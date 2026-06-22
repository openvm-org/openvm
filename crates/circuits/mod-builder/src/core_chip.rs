use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
    slice::from_raw_parts_mut,
    sync::Arc,
};

use itertools::Itertools;
use num_bigint::BigUint;
use num_traits::{ToPrimitive, Zero};
use openvm_circuit::{
    arch::*,
    system::memory::{online::TracingMemory, MemoryAuxColsFactory},
};
use openvm_circuit_primitives::{
    var_range::{SharedVariableRangeCheckerChip, VariableRangeCheckerChip},
    ColumnsAir, SubAir, TraceSubRowGenerator,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use crate::builder::{FieldExpr, FieldExprCols};

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

// No columns provided: wraps `FieldExpr`, whose column layout is built dynamically.
impl ColumnsAir for FieldExpressionCoreAir {}

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
            .map(|(flag, global_idx)| flag * AB::Expr::from_usize(global_idx))
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

pub struct FieldExpressionMetadata<F, A> {
    pub total_input_limbs: usize, // num_inputs * limbs_per_input
    _phantom: PhantomData<(F, A)>,
}

impl<F, A> Clone for FieldExpressionMetadata<F, A> {
    fn clone(&self) -> Self {
        Self {
            total_input_limbs: self.total_input_limbs,
            _phantom: PhantomData,
        }
    }
}

impl<F, A> Default for FieldExpressionMetadata<F, A> {
    fn default() -> Self {
        Self {
            total_input_limbs: 0,
            _phantom: PhantomData,
        }
    }
}

impl<F, A> FieldExpressionMetadata<F, A> {
    pub fn new(total_input_limbs: usize) -> Self {
        Self {
            total_input_limbs,
            _phantom: PhantomData,
        }
    }
}

impl<F, A> AdapterCoreMetadata for FieldExpressionMetadata<F, A>
where
    A: AdapterTraceExecutor<F>,
{
    #[inline(always)]
    fn get_adapter_width() -> usize {
        A::WIDTH * size_of::<F>()
    }
}

pub type FieldExpressionRecordLayout<F, A> = AdapterCoreLayout<FieldExpressionMetadata<F, A>>;

mod sealed {
    pub trait FieldExpressionLimbSealed {}

    impl FieldExpressionLimbSealed for u8 {}
    impl FieldExpressionLimbSealed for u16 {}
}

pub trait FieldExpressionLimb:
    sealed::FieldExpressionLimbSealed + Copy + Default + Send + Sync + 'static
{
    fn from_u64(value: u64) -> Self;
    fn as_u64(self) -> u64;
}

impl FieldExpressionLimb for u8 {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        value as u8
    }

    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

impl FieldExpressionLimb for u16 {
    #[inline(always)]
    fn from_u64(value: u64) -> Self {
        value as u16
    }

    #[inline(always)]
    fn as_u64(self) -> u64 {
        self as u64
    }
}

pub struct FieldExpressionCoreRecordMut<'a, T = u8> {
    pub opcode: &'a mut u8,
    pub input_limbs: &'a mut [T],
}

impl<'a, F, A, T>
    CustomBorrow<'a, FieldExpressionCoreRecordMut<'a, T>, FieldExpressionRecordLayout<F, A>>
    for [u8]
where
    T: FieldExpressionLimb,
{
    fn custom_borrow(
        &'a mut self,
        layout: FieldExpressionRecordLayout<F, A>,
    ) -> FieldExpressionCoreRecordMut<'a, T> {
        let opcode_size = size_of::<u8>().next_multiple_of(align_of::<T>());
        let (opcode_buf, input_limbs_buf) = self.split_at_mut(opcode_size);
        let opcode = &mut opcode_buf[0];
        let len = layout.metadata.total_input_limbs;
        // SAFETY: SizedRecord requests T alignment and enough bytes for len T limbs after opcode
        // padding.
        let input_limbs =
            unsafe { from_raw_parts_mut(input_limbs_buf.as_mut_ptr().cast::<T>(), len) };

        FieldExpressionCoreRecordMut {
            opcode,
            input_limbs,
        }
    }

    unsafe fn extract_layout(&self) -> FieldExpressionRecordLayout<F, A> {
        panic!("Should get the Layout information from FieldExpressionExecutor");
    }
}

impl<F, A, T> SizedRecord<FieldExpressionRecordLayout<F, A>> for FieldExpressionCoreRecordMut<'_, T>
where
    T: FieldExpressionLimb,
{
    fn size(layout: &FieldExpressionRecordLayout<F, A>) -> usize {
        size_of::<u8>().next_multiple_of(align_of::<T>())
            + layout.metadata.total_input_limbs * size_of::<T>()
    }

    fn alignment(_layout: &FieldExpressionRecordLayout<F, A>) -> usize {
        align_of::<T>()
    }
}

impl<'a, T> FieldExpressionCoreRecordMut<'a, T>
where
    T: FieldExpressionLimb,
{
    // This method is only used in testing
    pub fn new_from_execution_data(
        buffer: &'a mut [u8],
        inputs: &[BigUint],
        limbs_per_input: usize,
    ) -> Self {
        let record_info = FieldExpressionMetadata::<(), ()>::new(inputs.len() * limbs_per_input);

        let record: Self = buffer.custom_borrow(FieldExpressionRecordLayout {
            metadata: record_info,
        });
        record
    }

    #[inline(always)]
    pub fn fill_from_execution_data(&mut self, opcode: u8, data: &[T]) {
        // Rust will assert that length of `data` and `self.input_limbs` are the same
        // That is `data.len() == num_inputs * limbs_per_input`
        *self.opcode = opcode;
        self.input_limbs.copy_from_slice(data);
    }
}

#[derive(Clone)]
pub struct FieldExpressionExecutor<A> {
    adapter: A,
    pub expr: FieldExpr,
    pub offset: usize,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub name: String,
}

impl<A> FieldExpressionExecutor<A> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter: A,
        expr: FieldExpr,
        offset: usize,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        name: &str,
    ) -> Self {
        let opcode_flag_idx = if opcode_flag_idx.is_empty() && expr.needs_setup() {
            // single op chip that needs setup, so there is only one default flag, must be 0.
            vec![0]
        } else {
            // multi ops chip or no-setup chip, use as is.
            opcode_flag_idx
        };
        assert_eq!(opcode_flag_idx.len(), local_opcode_idx.len() - 1);
        tracing::debug!(
            "FieldExpressionCoreExecutor: opcode={name}, main_width={}",
            BaseAir::<BabyBear>::width(&expr)
        );
        Self {
            adapter,
            expr,
            offset,
            local_opcode_idx,
            opcode_flag_idx,
            name: name.to_string(),
        }
    }

    pub fn get_record_layout<F>(&self) -> FieldExpressionRecordLayout<F, A> {
        FieldExpressionRecordLayout {
            metadata: FieldExpressionMetadata::new(
                self.expr.builder.num_input * self.expr.canonical_num_limbs(),
            ),
        }
    }

    /// Returns a reference to the adapter for use in custom PreflightExecutor implementations.
    #[inline]
    pub fn adapter(&self) -> &A {
        &self.adapter
    }
}

pub struct FieldExpressionFiller<A, T = u8> {
    adapter: A,
    pub expr: FieldExpr,
    pub local_opcode_idx: Vec<usize>,
    pub opcode_flag_idx: Vec<usize>,
    pub range_checker: SharedVariableRangeCheckerChip,
    pub should_finalize: bool,
    _marker: PhantomData<T>,
}

impl<A, T> FieldExpressionFiller<A, T>
where
    T: FieldExpressionLimb,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        adapter: A,
        expr: FieldExpr,
        local_opcode_idx: Vec<usize>,
        opcode_flag_idx: Vec<usize>,
        range_checker: SharedVariableRangeCheckerChip,
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
        Self {
            adapter,
            expr,
            local_opcode_idx,
            opcode_flag_idx,
            range_checker,
            should_finalize,
            _marker: PhantomData,
        }
    }
    pub fn num_inputs(&self) -> usize {
        self.expr.builder.num_input
    }

    pub fn num_flags(&self) -> usize {
        self.expr.builder.num_flags
    }

    pub fn get_record_layout<F>(&self) -> FieldExpressionRecordLayout<F, A> {
        FieldExpressionRecordLayout {
            metadata: FieldExpressionMetadata::new(
                self.num_inputs() * self.expr.canonical_num_limbs(),
            ),
        }
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for FieldExpressionExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<F, ReadData: Into<DynArray<u8>>, WriteData: From<DynArray<u8>>>,
    for<'buf> RA: RecordArena<
        'buf,
        FieldExpressionRecordLayout<F, A>,
        (A::RecordMut<'buf>, FieldExpressionCoreRecordMut<'buf>),
    >,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, mut core_record) = state.ctx.alloc(self.get_record_layout());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let data: DynArray<_> = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        core_record.fill_from_execution_data(
            instruction.opcode.local_opcode_idx(self.offset) as u8,
            &data.0,
        );

        let (writes, _, _) = run_field_expression(
            &self.expr,
            &self.local_opcode_idx,
            &self.opcode_flag_idx,
            core_record.input_limbs,
            *core_record.opcode as usize,
        );

        self.adapter.write(
            state.memory,
            instruction,
            writes.into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }

    fn get_opcode_name(&self, _opcode: usize) -> String {
        self.name.clone()
    }
}

#[cfg(feature = "aot")]
impl<F: PrimeField32, A> AotExecutor<F> for FieldExpressionExecutor<A> {}

impl<F, A, T> TraceFiller<F> for FieldExpressionFiller<A, T>
where
    F: PrimeField32 + Send + Sync + Clone,
    A: 'static + AdapterTraceFiller<F>,
    T: FieldExpressionLimb,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // Get the core record from the row slice
        // SAFETY: Caller guarantees that row_slice has width A::WIDTH + core width
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };

        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY:
        // - caller ensures `core_row` contains a valid record representation that was previously
        //   written by the executor
        // - core_row slice is transmuted to FieldExpressionCoreRecordMut using the specified
        //   layout, which satisfies CustomBorrow requirements for safe access.
        let record: FieldExpressionCoreRecordMut<T> =
            unsafe { get_record_from_slice(&mut core_row, self.get_record_layout::<F>()) };

        let (_, inputs, flags) = run_field_expression(
            &self.expr,
            &self.local_opcode_idx,
            &self.opcode_flag_idx,
            record.input_limbs,
            *record.opcode as usize,
        );

        let range_checker = self.range_checker.as_ref();
        self.expr
            .generate_subrow((range_checker, inputs, flags), core_row);
    }

    fn fill_dummy_trace_row(&self, row_slice: &mut [F]) {
        if !self.should_finalize {
            return;
        }

        let inputs: Vec<BigUint> = vec![BigUint::zero(); self.num_inputs()];
        let flags: Vec<bool> = vec![false; self.num_flags()];
        let core_row = &mut row_slice[A::WIDTH..];
        // We **do not** want this trace row to update the range checker
        // so we must create a temporary range checker
        let tmp_range_checker = Arc::new(VariableRangeCheckerChip::new(self.range_checker.bus()));
        self.expr
            .generate_subrow((&tmp_range_checker, inputs, flags), core_row);
        core_row[0] = F::ZERO; // is_valid = 0
    }
}

fn limbs_to_biguint<T: FieldExpressionLimb>(limbs: &[T], limb_bits: usize) -> BigUint {
    let mut res = BigUint::zero();
    let base = BigUint::from(1u64 << limb_bits);
    for limb in limbs.iter().rev() {
        res = res * &base + BigUint::from(limb.as_u64());
    }
    res
}

fn biguint_to_limb_vec<T: FieldExpressionLimb>(
    mut value: BigUint,
    num_limbs: usize,
    limb_bits: usize,
) -> Vec<T> {
    let base = BigUint::from(1u64 << limb_bits);
    let mut out = Vec::with_capacity(num_limbs);
    for _ in 0..num_limbs {
        let limb = (&value % &base).to_u64().expect("limb should fit in u64");
        out.push(T::from_u64(limb));
        value /= &base;
    }
    out
}

fn flags_for_opcode(
    expr: &FieldExpr,
    local_opcode_flags: &[usize],
    opcode_flag_idx: &[usize],
    local_opcode_idx: usize,
) -> Vec<bool> {
    let mut flags = vec![];
    if expr.needs_setup() {
        flags = vec![false; expr.builder.num_flags];

        if let Some(opcode_position) = local_opcode_flags
            .iter()
            .position(|&idx| idx == local_opcode_idx)
        {
            if opcode_position < opcode_flag_idx.len() {
                let flag_idx = opcode_flag_idx[opcode_position];
                flags[flag_idx] = true;
            }
        }
    }
    flags
}

fn run_field_expression<T: FieldExpressionLimb>(
    expr: &FieldExpr,
    local_opcode_flags: &[usize],
    opcode_flag_idx: &[usize],
    data: &[T],
    local_opcode_idx: usize,
) -> (DynArray<T>, Vec<BigUint>, Vec<bool>) {
    let field_element_limbs = expr.canonical_num_limbs();
    let limb_bits = expr.canonical_limb_bits();
    assert_eq!(data.len(), expr.builder.num_input * field_element_limbs);

    let mut inputs = Vec::with_capacity(expr.builder.num_input);
    for i in 0..expr.builder.num_input {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        inputs.push(limbs_to_biguint(&data[start..end], limb_bits));
    }

    let flags = flags_for_opcode(expr, local_opcode_flags, opcode_flag_idx, local_opcode_idx);

    let vars = expr.execute(&inputs, &flags);
    assert_eq!(vars.len(), expr.builder.num_variables);

    let num_outputs = expr.builder.output_indices.len();
    let mut write_buffer = Vec::with_capacity(num_outputs * field_element_limbs);
    for &var_idx in expr.builder.output_indices.iter() {
        write_buffer.extend(biguint_to_limb_vec::<T>(
            vars[var_idx].clone(),
            field_element_limbs,
            limb_bits,
        ));
    }
    let writes: DynArray<_> = write_buffer.into();

    (writes, inputs, flags)
}

#[inline(always)]
pub fn run_field_expression_precomputed<const NEEDS_SETUP: bool>(
    expr: &FieldExpr,
    flag_idx: usize,
    data: &[u8],
) -> DynArray<u8> {
    run_field_expression_precomputed_bytes::<NEEDS_SETUP>(expr, flag_idx, data)
}

#[inline(always)]
pub fn run_field_expression_precomputed_bytes<const NEEDS_SETUP: bool>(
    expr: &FieldExpr,
    flag_idx: usize,
    data: &[u8],
) -> DynArray<u8> {
    let field_element_limbs = expr.canonical_num_limbs();
    let limb_bits = expr.canonical_limb_bits();
    assert_eq!(limb_bits % 8, 0);
    let bytes_per_limb = limb_bits / 8;
    let field_element_bytes = field_element_limbs * bytes_per_limb;
    assert_eq!(data.len(), expr.num_inputs() * field_element_bytes);

    let mut inputs = Vec::with_capacity(expr.num_inputs());
    for i in 0..expr.num_inputs() {
        let start = i * field_element_bytes;
        let end = start + field_element_bytes;
        inputs.push(BigUint::from_bytes_le(&data[start..end]));
    }

    let flags = if NEEDS_SETUP {
        let mut flags = vec![false; expr.num_flags()];
        if flag_idx < expr.num_flags() {
            flags[flag_idx] = true;
        }
        flags
    } else {
        vec![]
    };

    let vars = expr.execute(&inputs, &flags);
    assert_eq!(vars.len(), expr.num_vars());

    let mut write_buffer = vec![0u8; expr.output_indices().len() * field_element_bytes];
    for (i, &var_idx) in expr.output_indices().iter().enumerate() {
        let start = i * field_element_bytes;
        let bytes = vars[var_idx].to_bytes_le();
        let copy_len = bytes.len().min(field_element_bytes);
        write_buffer[start..start + copy_len].copy_from_slice(&bytes[..copy_len]);
    }
    write_buffer.into()
}

#[inline(always)]
pub fn run_field_expression_precomputed_limbs<const NEEDS_SETUP: bool, T: FieldExpressionLimb>(
    expr: &FieldExpr,
    flag_idx: usize,
    data: &[T],
) -> DynArray<T> {
    let field_element_limbs = expr.canonical_num_limbs();
    let limb_bits = expr.canonical_limb_bits();
    assert_eq!(data.len(), expr.num_inputs() * field_element_limbs);

    let mut inputs = Vec::with_capacity(expr.num_inputs());
    for i in 0..expr.num_inputs() {
        let start = i * field_element_limbs;
        let end = start + field_element_limbs;
        inputs.push(limbs_to_biguint(&data[start..end], limb_bits));
    }

    let flags = if NEEDS_SETUP {
        let mut flags = vec![false; expr.num_flags()];
        if flag_idx < expr.num_flags() {
            flags[flag_idx] = true;
        }
        flags
    } else {
        vec![]
    };

    let vars = expr.execute(&inputs, &flags);
    assert_eq!(vars.len(), expr.num_vars());

    let num_outputs = expr.output_indices().len();
    let mut write_buffer = Vec::with_capacity(num_outputs * field_element_limbs);
    for &var_idx in expr.output_indices().iter() {
        write_buffer.extend(biguint_to_limb_vec::<T>(
            vars[var_idx].clone(),
            field_element_limbs,
            limb_bits,
        ));
    }
    write_buffer.into()
}
