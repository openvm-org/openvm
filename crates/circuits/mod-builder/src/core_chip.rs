use crate::{
    builder::{FieldExpr, FieldExprCols},
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
};
use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterCoreLayout, AdapterExecutorE1, AdapterTraceFiller,
        AdapterTraceStep, CustomBorrow, DynAdapterInterface, DynArray, MinimalInstruction,
        RecordArena, Result, StepExecutorE1, TraceFiller, TraceStep, VmAdapterInterface, VmCoreAir,
        VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::SharedVariableRangeCheckerChip, AlignedBytesBorrow, SubAir, TraceSubRowGenerator,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use std::{borrow::BorrowMut, mem::size_of};
#[derive(Clone)]
pub struct FieldExpressionCoreAir {
    pub expr: FieldExpr,

    /// The global opcode offset.
    pub offset: usize,

    /// All the opcode indices (including setup) supported by this Air.
    /// The last one must be the setup opcode if it's a chip needs setup.
    pub local_opcode_idx: Vec<usize>,
    /// Opcode flag idx (indices from builder.new_flag()) for all except setup opcode. Empty if
    /// single op chip.
    pub opcode_flag_idx: Vec<usize>,
    // Example 1: 1-op chip EcAdd that needs setup
    //   local_opcode_idx = [0, 2], where 0 is EcAdd, 2 is setup
    //   opcode_flag_idx = [], not needed for single op chip.
    // Example 2: 1-op chip EvaluateLine that doesn't need setup
    //   local_opcode_idx = [2], the id within PairingOpcodeEnum
    //   opcode_flag_idx = [], not needed
    // Example 3: 2-op chip MulDiv that needs setup
    //   local_opcode_idx = [2, 3, 4], where 2 is Mul, 3 is Div, 4 is setup
    //   opcode_flag_idx = [0, 1], where 0 is mul_flag, 1 is div_flag, in the builder
    // We don't support 2-op chip that doesn't need setup right now.
}

impl FieldExpressionCoreAir {
    pub fn new(
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
    ) -> Self {
        let opcode_flag_idx = if opcode_flag_idx.is_empty() && expr.needs_setup() {
            // single op chip that needs setup, so there is only one default flag, must be 0.
            vec![0]
        } else {
            // multi ops chip or no-setup chip, use as is.
            opcode_flag_idx
        };
        assert_eq!(opcode_flag_idx.len(), local_opcode_idx.len() - 1);
        Self {
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.expr.builder.num_input
    }

    pub fn num_vars(&self) -> usize {
        self.expr.builder.num_variables
    }

    pub fn num_flags(&self) -> usize {
        self.expr.builder.num_flags
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.expr.builder.output_indices
    }
}

impl<F: Field> BaseAir<F> for FieldExpressionCoreAir {
    fn width(&self) -> usize {
        BaseAir::<F>::width(&self.expr)
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FieldExpressionCoreAir {}

impl<AB: InteractionBuilder, I> VmCoreAir<AB, I> for FieldExpressionCoreAir
where
    I: VmAdapterInterface<AB::Expr>,
    AdapterAirContext<AB::Expr, I>:
        From<AdapterAirContext<AB::Expr, DynAdapterInterface<AB::Expr>>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        assert_eq!(local.len(), BaseAir::<AB::F>::width(&self.expr));
        self.expr.eval(builder, local);
        let FieldExprCols {
            is_valid,
            inputs,
            vars,
            flags,
            ..
        } = self.expr.load_vars(local);
        assert_eq!(inputs.len(), self.num_inputs());
        assert_eq!(vars.len(), self.num_vars());
        assert_eq!(flags.len(), self.num_flags());
        let reads: Vec<AB::Expr> = inputs.concat().iter().map(|x| (*x).into()).collect();
        let writes: Vec<AB::Expr> = self
            .output_indices()
            .iter()
            .flat_map(|&i| vars[i].clone())
            .map(Into::into)
            .collect();

        let opcode_flags_except_last = self.opcode_flag_idx.iter().map(|&i| flags[i]).collect_vec();
        let last_opcode_flag = is_valid
            - opcode_flags_except_last
                .iter()
                .map(|&v| v.into())
                .sum::<AB::Expr>();
        builder.assert_bool(last_opcode_flag.clone());
        let opcode_flags = opcode_flags_except_last
            .into_iter()
            .map(Into::into)
            .chain(Some(last_opcode_flag));
        let expected_opcode = opcode_flags
            .zip(self.local_opcode_idx.iter().map(|&i| i + self.offset))
            .map(|(flag, global_idx)| flag * AB::Expr::from_canonical_usize(global_idx))
            .sum();

        let instruction = MinimalInstruction {
            is_valid: is_valid.into(),
            opcode: expected_opcode,
        };

        let ctx: AdapterAirContext<_, DynAdapterInterface<_>> = AdapterAirContext {
            to_pc: None,
            reads: reads.into(),
            writes: writes.into(),
            instruction: instruction.into(),
        };
        ctx.into()
    }

    fn start_offset(&self) -> usize {
        self.offset
    }
}

// Metadata record
#[derive(Clone)]
pub struct FieldExpressionMetadata {
    pub total_input_limbs: usize, // num_inputs * limbs_per_input
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FieldExpressionCoreRecord {
    pub opcode: u8,
}

pub struct FieldExpressionCoreRecordMut<'a> {
    pub inner: &'a mut FieldExpressionCoreRecord,
    pub input_limbs: &'a mut [u8],
}

impl<'a> CustomBorrow<'a, FieldExpressionCoreRecordMut<'a>, FieldExpressionMetadata> for [u8] {
    fn custom_borrow(
        &'a mut self,
        metadata: FieldExpressionMetadata,
    ) -> FieldExpressionCoreRecordMut<'a> {
        let (record_buf, input_limbs_buf) =
            unsafe { self.split_at_mut_unchecked(size_of::<FieldExpressionCoreRecord>()) };

        FieldExpressionCoreRecordMut {
            inner: record_buf.borrow_mut(),
            input_limbs: &mut input_limbs_buf[..metadata.total_input_limbs],
        }
    }
}

impl<'a> FieldExpressionCoreRecordMut<'a> {
    pub fn new_from_execution_data(
        buffer: &'a mut [u8],
        _opcode: u8,
        inputs: &[BigUint],
        _limb_bits: usize,
        limbs_per_input: usize,
    ) -> Self {
        let record_info = FieldExpressionMetadata {
            total_input_limbs: inputs.len() * limbs_per_input,
        };

        let record: Self = buffer.custom_borrow(record_info);
        // record.fill_from_execution_data(opcode, );
        record
    }

    #[inline(always)]
    pub fn fill_from_execution_data(&mut self, opcode: u8, data: &[u8]) {
        // Rust will assert that that length of `data` and `self.input_limbs` are the same
        // That is `data.len() == num_inputs * limbs_per_input`
        self.inner.opcode = opcode;
        self.input_limbs.copy_from_slice(data);
    }
}

// TODO(arayi): use lifetimes and references for fields
pub struct FieldExpressionStep<A> {
    adapter: A,
    pub expr: FieldExpr,
    pub offset: usize,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub name: String,
    pub should_finalize: bool,
}

impl<A> FieldExpressionStep<A> {
    pub fn new(
        adapter: A,
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        range_checker: SharedVariableRangeCheckerChip,
        name: &str,
        should_finalize: bool,
    ) -> Self {
        let opcode_flag_idx = if opcode_flag_idx.is_empty() && expr.needs_setup() {
            // single op chip that needs setup, so there is only one default flag, must be 0.
            vec![0]
        } else {
            // multi ops chip or no-setup chip, use as is.
            opcode_flag_idx
        };
        assert_eq!(opcode_flag_idx.len(), local_opcode_idx.len() - 1);
        tracing::info!(
            "FieldExpressionCoreStep: opcode={name}, main_width={}",
            BaseAir::<BabyBear>::width(&expr)
        );
        Self {
            adapter,
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            name: name.to_string(),
            should_finalize,
        }
    }
    pub fn num_inputs(&self) -> usize {
        self.expr.builder.num_input
    }

    pub fn num_vars(&self) -> usize {
        self.expr.builder.num_variables
    }

    pub fn num_flags(&self) -> usize {
        self.expr.builder.num_flags
    }

    pub fn output_indices(&self) -> &[usize] {
        &self.expr.builder.output_indices
    }
}

impl<F, CTX, A> TraceStep<F, CTX> for FieldExpressionStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterTraceStep<
            F,
            CTX,
            ReadData: Into<DynArray<u8>>,
            WriteData: From<DynArray<u8>>,
        >,
{
    type RecordLayout = AdapterCoreLayout<FieldExpressionMetadata>;
    type RecordMut<'a> = (A::RecordMut<'a>, FieldExpressionCoreRecordMut<'a>);

    fn execute<'buf, RA>(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        arena: &'buf mut RA,
    ) -> Result<()>
    where
        RA: RecordArena<'buf, Self::RecordLayout, Self::RecordMut<'buf>>,
    {
        let core_record_metadata = FieldExpressionMetadata {
            total_input_limbs: self.num_inputs() * self.expr.canonical_num_limbs(),
        };

        let (mut adapter_record, mut core_record) = arena.alloc(AdapterCoreLayout::with_metadata(
            A::WIDTH,
            core_record_metadata,
        ));

        A::start(*state.pc, state.memory, &mut adapter_record);

        let data: DynArray<_> = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.fill_from_execution_data(
            instruction.opcode.local_opcode_idx(self.offset) as u8,
            &data.0,
        );

        let (writes, _, _) = run_field_expression(self, &data.0, core_record.inner.opcode as usize);

        self.adapter.write(
            state.memory,
            instruction,
            &writes.into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.name.clone()
    }
}

impl<F, CTX, A> TraceFiller<F, CTX> for FieldExpressionStep<A>
where
    F: PrimeField32 + Send + Sync + Clone,
    A: 'static + AdapterTraceFiller<F, CTX>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // NOTE[teokitan]: This is where GPU acceleration should happen in the future

        // Get the core record from the row slice
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, adapter_row);

        let core_bytes = unsafe {
            std::slice::from_raw_parts_mut(
                core_row.as_mut_ptr() as *mut u8,
                core_row.len() * size_of::<F>(),
            )
        };

        let record_metadata = FieldExpressionMetadata {
            total_input_limbs: self.num_inputs() * self.expr.canonical_num_limbs(),
        };

        let record: FieldExpressionCoreRecordMut = core_bytes.custom_borrow(record_metadata);

        let (_, inputs, flags) =
            run_field_expression(self, &record.input_limbs, record.inner.opcode as usize);

        let range_checker = self.range_checker.as_ref();
        self.expr
            .generate_subrow((range_checker, inputs, flags), core_row);
    }

    fn fill_dummy_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        if !self.should_finalize {
            return;
        }

        let inputs: Vec<BigUint> = vec![BigUint::zero(); self.num_inputs()];
        let flags: Vec<bool> = vec![false; self.num_flags()];
        let core_row = &mut row_slice[A::WIDTH..];
        // We **do not** want this trace row to update the range checker
        // so we must create a temporary range checker
        let tmp_range_checker = SharedVariableRangeCheckerChip::new(self.range_checker.bus());
        self.expr
            .generate_subrow((tmp_range_checker.as_ref(), inputs, flags), core_row);
        core_row[0] = F::ZERO; // is_valid = 0
    }
}

impl<F, A> StepExecutorE1<F> for FieldExpressionStep<A>
where
    F: PrimeField32,
    A: 'static
        + for<'a> AdapterExecutorE1<F, ReadData: Into<DynArray<u8>>, WriteData: From<DynArray<u8>>>,
{
    fn execute_e1<Ctx>(
        &self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let data: DynArray<_> = self.adapter.read(state, instruction).into();

        let writes = run_field_expression(
            self,
            &data.0,
            instruction.opcode.local_opcode_idx(self.offset) as usize,
        )
        .0;
        self.adapter.write(state, instruction, &writes.into());
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn execute_metered(
        &self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += 1;

        Ok(())
    }
}

fn run_field_expression<A>(
    step: &FieldExpressionStep<A>,
    data: &[u8],
    local_opcode_idx: usize,
) -> (DynArray<u8>, Vec<BigUint>, Vec<bool>) {
    let field_element_limbs = step.expr.canonical_num_limbs();
    let limb_bits = step.expr.canonical_limb_bits();

    let data = data.iter().map(|&x| x as u32).collect_vec();

    assert_eq!(data.len(), step.num_inputs() * field_element_limbs);

    let mut inputs = Vec::with_capacity(step.num_inputs());
    for i in 0..step.num_inputs() {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        let limb_slice = &data[start..end];
        let input = limbs_to_biguint(limb_slice, limb_bits);
        inputs.push(input);
    }

    // Reconstruct flags from opcode
    let mut flags = vec![];
    if step.expr.needs_setup() {
        flags = vec![false; step.num_flags()];

        // Find which opcode this is in our local_opcode_idx list
        if let Some(opcode_position) = step
            .local_opcode_idx
            .iter()
            .position(|&idx| idx == local_opcode_idx)
        {
            // If this is NOT the last opcode (setup), set the corresponding flag
            if opcode_position < step.opcode_flag_idx.len() {
                let flag_idx = step.opcode_flag_idx[opcode_position];
                flags[flag_idx] = true;
            }
            // If opcode_position == step.opcode_flag_idx.len(), it's the setup operation
            // and all flags should remain false (which they already are)
        }
    }

    let vars = step.expr.execute(inputs.clone(), flags.clone());
    assert_eq!(vars.len(), step.num_vars());

    let outputs: Vec<BigUint> = step
        .output_indices()
        .iter()
        .map(|&i| vars[i].clone())
        .collect();
    let writes: DynArray<_> = outputs
        .iter()
        .map(|x| biguint_to_limbs_vec(x.clone(), limb_bits, field_element_limbs))
        .concat()
        .into_iter()
        .map(|x| x as u8)
        .collect::<Vec<_>>()
        .into();

    (writes, inputs, flags)
}
