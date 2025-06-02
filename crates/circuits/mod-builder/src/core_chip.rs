use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::Zero;
use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        AdapterAirContext, AdapterExecutorE1, AdapterTraceStep, DynAdapterInterface, DynArray,
        MinimalInstruction, Result, StepExecutorE1, TraceStep, VmAdapterInterface, VmCoreAir,
        VmStateMut,
    },
    system::memory::{
        online::{GuestMemory, TracingMemory},
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{
    var_range::SharedVariableRangeCheckerChip, SubAir, TraceSubRowGenerator,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, FieldAlgebra, PrimeField32},
    rap::BaseAirWithPublicValues,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::{
    utils::{biguint_to_limbs_vec, limbs_to_biguint},
    FieldExpr, FieldExprCols,
};

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
            TraceContext<'a> = (),
        >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.name.clone()
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        width: usize,
    ) -> Result<()> {
        let row_slice = &mut trace[*trace_offset..*trace_offset + width];
        let (adapter_row, core_row) = row_slice.split_at_mut(A::WIDTH);

        A::start(*state.pc, state.memory, adapter_row);

        let data: DynArray<_> = self
            .adapter
            .read(state.memory, instruction, adapter_row)
            .into();

        let (writes, inputs, flags) = run_field_expression(self, &data, instruction);

        // TODO(arayi): Should move this to fill_trace_row
        self.expr
            .generate_subrow((self.range_checker.as_ref(), inputs, flags), core_row);

        self.adapter
            .write(state.memory, instruction, adapter_row, &writes.into());

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        *trace_offset += width;
        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row: &mut [F]) {
        let (adapter_row, _) = row.split_at_mut(A::WIDTH);
        self.adapter.fill_trace_row(mem_helper, (), adapter_row);
    }

    // We will be setting is_valid = 0. That forces all flags be 0 (otherwise setup will be -1).
    // We generate a dummy row with all flags set to 0, then we set is_valid = 0.
    fn fill_dummy_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, row: &mut [F]) {
        if !self.should_finalize {
            return;
        }
        let inputs: Vec<BigUint> = vec![BigUint::zero(); self.num_inputs()];
        let flags: Vec<bool> = vec![false; self.num_flags()];
        let core_row = &mut row[A::WIDTH..];
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

        let writes = run_field_expression(self, &data, instruction).0;
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

fn run_field_expression<F: PrimeField32, A>(
    step: &FieldExpressionStep<A>,
    data: &DynArray<u8>,
    instruction: &Instruction<F>,
) -> (DynArray<u8>, Vec<BigUint>, Vec<bool>) {
    let field_element_limbs = step.expr.canonical_num_limbs();
    let limb_bits = step.expr.canonical_limb_bits();

    let data = data.0.iter().map(|&x| x as u32).collect_vec();

    assert_eq!(data.len(), step.num_inputs() * field_element_limbs);

    let mut inputs = Vec::with_capacity(step.num_inputs());
    for i in 0..step.num_inputs() {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        let limb_slice = &data[start..end];
        let input = limbs_to_biguint(limb_slice, limb_bits);
        inputs.push(input);
    }

    let Instruction { opcode, .. } = instruction;
    let local_opcode_idx = opcode.local_opcode_idx(step.offset);
    let mut flags = vec![];

    // If the chip doesn't need setup, (right now) it must be single op chip and thus no flag is
    // needed. Otherwise, there is a flag for each opcode and will be derived by
    // is_valid - sum(flags).
    if step.expr.needs_setup() {
        flags = vec![false; step.num_flags()];
        step.opcode_flag_idx
            .iter()
            .enumerate()
            .for_each(|(i, &flag_idx)| {
                flags[flag_idx] = local_opcode_idx == step.local_opcode_idx[i]
            });
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
