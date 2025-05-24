use core::ops::Deref;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::offset_of,
    sync::{Arc, Mutex},
};

use itertools::zip_eq;
use openvm_circuit::{
    arch::{
        execution_mode::{metered::MeteredCtx, E1E2ExecutionCtx},
        ExecutionBridge, ExecutionState, NewVmChipWrapper, Result, StepExecutorE1, Streams,
        TraceStep, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols, AUX_LEN},
        online::{GuestMemory, TracingMemory},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::is_less_than::LessThanAuxCols;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FriOpcode::FRI_REDUCED_OPENING};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use static_assertions::const_assert_eq;

use crate::{
    adapters::{
        memory_read_native, memory_write_native, tracing_read_native, tracing_write_native,
    },
    field_extension::{FieldExtension, EXT_DEG},
    utils::const_max,
};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct WorkloadCols<T> {
    prefix: PrefixCols<T>,

    a_aux: MemoryWriteAuxCols<T, 1>,
    /// The value of `b` read.
    b: [T; EXT_DEG],
    b_aux: MemoryReadAuxCols<T>,
}
const WL_WIDTH: usize = WorkloadCols::<u8>::width();
const_assert_eq!(WL_WIDTH, 27);

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct Instruction1Cols<T> {
    prefix: PrefixCols<T>,

    pc: T,

    a_ptr_ptr: T,
    a_ptr_aux: MemoryReadAuxCols<T>,

    b_ptr_ptr: T,
    b_ptr_aux: MemoryReadAuxCols<T>,

    /// Extraneous column that is constrained to write_a * a_or_is_first but has no meaningful
    /// effect. It can be removed along with its constraints without impacting correctness.
    write_a_x_is_first: T,
}
const INS_1_WIDTH: usize = Instruction1Cols::<u8>::width();
const_assert_eq!(INS_1_WIDTH, 26);
const_assert_eq!(
    offset_of!(WorkloadCols<u8>, prefix),
    offset_of!(Instruction1Cols<u8>, prefix)
);

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct Instruction2Cols<T> {
    general: GeneralCols<T>,
    /// Shared with `a_or_is_first` in other column types. Must be 0 for Instruction2Cols.
    is_first: T,

    length_ptr: T,
    length_aux: MemoryReadAuxCols<T>,

    alpha_ptr: T,
    alpha_aux: MemoryReadAuxCols<T>,

    result_ptr: T,
    result_aux: MemoryWriteAuxCols<T, EXT_DEG>,

    hint_id_ptr: T,

    is_init_ptr: T,
    is_init_aux: MemoryReadAuxCols<T>,

    /// Extraneous column that is constrained to write_a * a_or_is_first but has no meaningful
    /// effect. It can be removed along with its constraints without impacting correctness.
    write_a_x_is_first: T,
}
const INS_2_WIDTH: usize = Instruction2Cols::<u8>::width();
const_assert_eq!(INS_2_WIDTH, 26);
const_assert_eq!(
    offset_of!(WorkloadCols<u8>, prefix) + offset_of!(PrefixCols<u8>, general),
    offset_of!(Instruction2Cols<u8>, general)
);
const_assert_eq!(
    offset_of!(Instruction1Cols<u8>, prefix) + offset_of!(PrefixCols<u8>, a_or_is_first),
    offset_of!(Instruction2Cols<u8>, is_first)
);
const_assert_eq!(
    offset_of!(Instruction1Cols<u8>, write_a_x_is_first),
    offset_of!(Instruction2Cols<u8>, write_a_x_is_first)
);

pub const OVERALL_WIDTH: usize = const_max(const_max(WL_WIDTH, INS_1_WIDTH), INS_2_WIDTH);
const_assert_eq!(OVERALL_WIDTH, 27);

/// Every row starts with these columns.
#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct GeneralCols<T> {
    /// Whether the row is a workload row.
    is_workload_row: T,
    /// Whether the row is an instruction row.
    is_ins_row: T,
    /// For Instruction1 rows, the initial timestamp of the FRI_REDUCED_OPENING instruction.
    /// For Workload rows, the final timestamp after processing the next elements minus
    /// `INSTRUCTION_READS`. For Instruction2 rows, unused.
    timestamp: T,
}
const GENERAL_WIDTH: usize = GeneralCols::<u8>::width();
const_assert_eq!(GENERAL_WIDTH, 3);

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct DataCols<T> {
    /// For Instruction1 rows, `mem[a_ptr_ptr]`.
    /// For Workload rows, the pointer in a-values after increment.
    a_ptr: T,
    /// Indicates whether to write a-value or read it.
    /// For Instruction1 rows, `1 - mem[is_init_ptr]`.
    /// For Workload rows, whether we are writing the a-value or reading it; fixed for entire
    /// workload/instruction block.
    write_a: T,
    /// For Instruction1 rows, `mem[b_ptr_ptr]`.
    /// For Workload rows, the pointer in b-values after increment.
    b_ptr: T,
    /// For Instruction1 rows, the value read from `mem[length_ptr]`.
    /// For Workload rows, the workload row index from the top. *Not* the index into a-values and
    /// b-values. (Note: idx increases within a workload/instruction block, while timestamp, a_ptr,
    /// and b_ptr decrease.)
    idx: T,
    /// For both Instruction1 and Workload rows, equal to sum_{k=0}^{idx} alpha^{len-i} (b_i -
    /// a_i). Instruction1 rows constrain this to be the result written to `mem[result_ptr]`.
    result: [T; EXT_DEG],
    /// The alpha to use in this instruction. Fixed across workload rows; Instruction1 rows read
    /// this from `mem[alpha_ptr]`.
    alpha: [T; EXT_DEG],
}
#[allow(dead_code)]
const DATA_WIDTH: usize = DataCols::<u8>::width();
const_assert_eq!(DATA_WIDTH, 12);

/// Prefix of `WorkloadCols` and `Instruction1Cols`
#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct PrefixCols<T> {
    general: GeneralCols<T>,
    /// WorkloadCols uses this column as the value of `a` read. Instruction1Cols uses this column
    /// as the `is_first` flag must be set to one. Shared with Instruction2Cols `is_first`.
    a_or_is_first: T,
    data: DataCols<T>,
}
const PREFIX_WIDTH: usize = PrefixCols::<u8>::width();
const_assert_eq!(PREFIX_WIDTH, 16);

const INSTRUCTION_READS: usize = 5;

/// A valid trace is a sequence of blocks, where each block has the following consecutive rows:
/// 1. **Workload Columns**: A sequence of rows used to compute the "rolling hash" of b - a.
/// 2. **Instruction1**: The "local" row for the instruction window.
/// 3. **Instruction2**: The "next" row for the instruction window.
///
/// The row mode is determined by the following flags:
/// * `GeneralCols.is_workload_row`: Indicator for a Workload row.
/// * `GeneralCols.is_ins_row`: Indicator for an Instruction1 or Instruction2 row.
/// * `PrefixCols.a_or_is_first` / `Instruction2Cols.is_first`: For Instruction1 or Instruction2
///   rows, indicator for Instruction1 rows.
///
/// We impose the following flag constraints:
/// * (F1): Every row is either a Workload row, an Instruction row, or Disabled.
///
/// A trace may also end in one or more Disabled rows, which emit no interactions and for which
/// the all-zeroes row is valid.
///
/// The AIR enforces the following transitions, which define the block structure outlined above:
/// * (T1): The trace must start with a Workload row or a Disabled row.
/// * (T2): A Disabled row can only be followed by a Disabled row (except on last).
/// * (T3): A Workload row cannot be followed by a Disabled row.
/// * (T4): A non-Instruction must not be followed by an Instruction2 row.
/// * (T5): An Instruction1 row must be followed by an Instruction2 row.
/// * (T6): An Instruction2 row can only be followed by a Workload or Disabled row.
/// * (T7): The last row is either a Disabled or an Instruction2 row.
///
/// Note that (T2) + (T3) + (T4) together imply that a Workload row can only be followed by an
/// Instruction1 row, as desired.
///
/// Note that these transition constraints do allow for a somewhat degenerate trace consisting of
/// only disabled rows. If the trace does not have this degenerate form, then the constraints ensure
/// it starts with a Workload row (T1) and ends with either a Disabled or Instruction2 row (T7).
/// The other transition constraints then ensure the proper state transitions from Workload to
/// Instruction2.
#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct FriReducedOpeningAir {
    execution_bridge: ExecutionBridge,
    memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for FriReducedOpeningAir {
    fn width(&self) -> usize {
        OVERALL_WIDTH
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FriReducedOpeningAir {}
impl<F: Field> PartitionedBaseAir<F> for FriReducedOpeningAir {}
impl<AB: InteractionBuilder> Air<AB> for FriReducedOpeningAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);
        let local_slice = local.deref();
        let next_slice = next.deref();
        self.eval_general(builder, local_slice, next_slice);
        self.eval_workload_row(builder, local_slice, next_slice);
        self.eval_instruction1_row(builder, local_slice, next_slice);
        self.eval_instruction2_row(builder, local_slice, next_slice);
    }
}

impl FriReducedOpeningAir {
    fn eval_general<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_slice: &[AB::Var],
        next_slice: &[AB::Var],
    ) {
        let local: &GeneralCols<AB::Var> = local_slice[..GENERAL_WIDTH].borrow();
        let next: &GeneralCols<AB::Var> = next_slice[..GENERAL_WIDTH].borrow();
        // (F1): Every row is either a Workload row, an Instruction row, or Disabled.
        {
            builder.assert_bool(local.is_ins_row);
            builder.assert_bool(local.is_workload_row);
            builder.assert_bool(local.is_ins_row + local.is_workload_row);
        }
        //  (T2): A Disabled row can only be followed by a Disabled row (except on last).
        {
            let mut when_transition = builder.when_transition();
            let mut when_disabled =
                when_transition.when_ne(local.is_ins_row + local.is_workload_row, AB::Expr::ONE);
            when_disabled.assert_zero(next.is_ins_row + next.is_workload_row);
        }
    }

    fn eval_workload_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_slice: &[AB::Var],
        next_slice: &[AB::Var],
    ) {
        let local: &WorkloadCols<AB::Var> = local_slice[..WL_WIDTH].borrow();
        let next: &PrefixCols<AB::Var> = next_slice[..PREFIX_WIDTH].borrow();
        let local_data = &local.prefix.data;
        let start_timestamp = next.general.timestamp;
        let multiplicity = local.prefix.general.is_workload_row;
        // a_ptr/b_ptr/length/result
        let ptr_reads = AB::F::from_canonical_usize(INSTRUCTION_READS);
        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);
        // write_a itself could be anything on non-workload row, but on workload row, it must be
        // boolean. write_a on last workflow row will be constrained to equal write_a on
        // instruction1 row, implying the latter is boolean.
        builder.when(multiplicity).assert_bool(local_data.write_a);
        // read a when write_a is 0
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), next.data.a_ptr),
                [local.prefix.a_or_is_first],
                start_timestamp + ptr_reads,
                local.a_aux.as_ref(),
            )
            .eval(builder, (AB::Expr::ONE - local_data.write_a) * multiplicity);
        // write a when write_a is 1
        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), next.data.a_ptr),
                [local.prefix.a_or_is_first],
                start_timestamp + ptr_reads,
                &local.a_aux,
            )
            .eval(builder, local_data.write_a * multiplicity);
        // read b
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), next.data.b_ptr),
                local.b,
                start_timestamp + ptr_reads + AB::Expr::ONE,
                &local.b_aux,
            )
            .eval(builder, multiplicity);
        {
            let mut when_transition = builder.when_transition();
            let mut builder = when_transition.when(local.prefix.general.is_workload_row);
            // ATTENTION: degree of builder is 2
            // local.timestamp = next.timestamp + 2
            builder.assert_eq(
                local.prefix.general.timestamp,
                start_timestamp + AB::Expr::TWO,
            );
            // local.idx = next.idx + 1
            builder.assert_eq(local_data.idx + AB::Expr::ONE, next.data.idx);
            // local.alpha = next.alpha
            assert_array_eq(&mut builder, local_data.alpha, next.data.alpha);
            // local.a_ptr = next.a_ptr + 1
            builder.assert_eq(local_data.a_ptr, next.data.a_ptr + AB::F::ONE);
            // local.write_a = next.write_a
            builder.assert_eq(local_data.write_a, next.data.write_a);
            // local.b_ptr = next.b_ptr + EXT_DEG
            builder.assert_eq(
                local_data.b_ptr,
                next.data.b_ptr + AB::F::from_canonical_usize(EXT_DEG),
            );
            // local.timestamp = next.timestamp + 2
            builder.assert_eq(
                local.prefix.general.timestamp,
                next.general.timestamp + AB::Expr::TWO,
            );
            // local.result * local.alpha + local.b - local.a = next.result
            let mut expected_result = FieldExtension::multiply(local_data.result, local_data.alpha);
            expected_result
                .iter_mut()
                .zip(local.b.iter())
                .for_each(|(e, b)| {
                    *e += (*b).into();
                });
            expected_result[0] -= local.prefix.a_or_is_first.into();
            assert_array_eq(&mut builder, expected_result, next.data.result);
        }
        {
            let mut next_ins = builder.when(next.general.is_ins_row);
            let mut local_non_ins =
                next_ins.when_ne(local.prefix.general.is_ins_row, AB::Expr::ONE);
            // (T4): A non-Instruction must not be followed by an Instruction2 row.
            local_non_ins.assert_one(next.a_or_is_first);

            // (T3): A Workload row cannot be followed by a Disabled row.
            builder
                .when(local.prefix.general.is_workload_row)
                .assert_one(next.general.is_ins_row + next.general.is_workload_row);
        }
        {
            let mut when_first_row = builder.when_first_row();
            let mut when_enabled = when_first_row
                .when(local.prefix.general.is_ins_row + local.prefix.general.is_workload_row);
            // (T1): The trace must start with a Workload row or a Disabled row.
            when_enabled.assert_one(local.prefix.general.is_workload_row);
            // Workload rows must start with the first element.
            when_enabled.assert_zero(local.prefix.data.idx);
            // local.result is all 0s.
            assert_array_eq(
                &mut when_enabled,
                local.prefix.data.result,
                [AB::Expr::ZERO; EXT_DEG],
            );
        }
    }

    fn eval_instruction1_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_slice: &[AB::Var],
        next_slice: &[AB::Var],
    ) {
        let local: &Instruction1Cols<AB::Var> = local_slice[..INS_1_WIDTH].borrow();
        let next: &Instruction2Cols<AB::Var> = next_slice[..INS_2_WIDTH].borrow();
        // `is_ins_row` already indicates enabled.
        let mut is_ins_row = builder.when(local.prefix.general.is_ins_row);
        // These constraints do not add anything and can be safely removed.
        {
            is_ins_row.assert_eq(
                local.write_a_x_is_first,
                local.prefix.data.write_a * local.prefix.a_or_is_first,
            );
            is_ins_row.assert_bool(local.write_a_x_is_first);
        }
        let mut is_first_ins = is_ins_row.when(local.prefix.a_or_is_first);
        // ATTENTION: degree of is_first_ins is 2
        // (T5): An Instruction1 row must be followed by an Instruction2 row.
        {
            is_first_ins.assert_one(next.general.is_ins_row);
            is_first_ins.assert_zero(next.is_first);
        }

        let local_data = &local.prefix.data;
        let length = local.prefix.data.idx;
        let multiplicity = local.prefix.general.is_ins_row * local.prefix.a_or_is_first;
        let start_timestamp = local.prefix.general.timestamp;
        let write_timestamp = start_timestamp
            + AB::Expr::TWO * length
            + AB::Expr::from_canonical_usize(INSTRUCTION_READS);
        let end_timestamp = write_timestamp.clone() + AB::Expr::ONE;
        let native_as = AB::Expr::from_canonical_u32(AS::Native as u32);
        self.execution_bridge
            .execute(
                AB::F::from_canonical_usize(FRI_REDUCED_OPENING.global_opcode().as_usize()),
                [
                    local.a_ptr_ptr.into(),
                    local.b_ptr_ptr.into(),
                    next.length_ptr.into(),
                    next.alpha_ptr.into(),
                    next.result_ptr.into(),
                    next.hint_id_ptr.into(),
                    next.is_init_ptr.into(),
                ],
                ExecutionState::new(local.pc, local.prefix.general.timestamp),
                ExecutionState::<AB::Expr>::new(
                    AB::Expr::from_canonical_u32(DEFAULT_PC_STEP) + local.pc,
                    end_timestamp.clone(),
                ),
            )
            .eval(builder, multiplicity.clone());
        // Read alpha
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), next.alpha_ptr),
                local_data.alpha,
                start_timestamp,
                &next.alpha_aux,
            )
            .eval(builder, multiplicity.clone());
        // Read length.
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), next.length_ptr),
                [length],
                start_timestamp + AB::Expr::ONE,
                &next.length_aux,
            )
            .eval(builder, multiplicity.clone());
        // Read a_ptr
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), local.a_ptr_ptr),
                [local_data.a_ptr],
                start_timestamp + AB::Expr::TWO,
                &local.a_ptr_aux,
            )
            .eval(builder, multiplicity.clone());
        // Read b_ptr
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), local.b_ptr_ptr),
                [local_data.b_ptr],
                start_timestamp + AB::Expr::from_canonical_u32(3),
                &local.b_ptr_aux,
            )
            .eval(builder, multiplicity.clone());
        // Read write_a = 1 - is_init, it should be a boolean.
        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), next.is_init_ptr),
                [AB::Expr::ONE - local_data.write_a],
                start_timestamp + AB::Expr::from_canonical_u32(4),
                &next.is_init_aux,
            )
            .eval(builder, multiplicity.clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), next.result_ptr),
                local_data.result,
                write_timestamp,
                &next.result_aux,
            )
            .eval(builder, multiplicity.clone());
    }

    fn eval_instruction2_row<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_slice: &[AB::Var],
        next_slice: &[AB::Var],
    ) {
        let local: &Instruction2Cols<AB::Var> = local_slice[..INS_2_WIDTH].borrow();
        let next: &WorkloadCols<AB::Var> = next_slice[..WL_WIDTH].borrow();
        // (T7): The last row is either a Disabled or an Instruction2 row.
        {
            let mut last_row = builder.when_last_row();
            let mut enabled =
                last_row.when(local.general.is_ins_row + local.general.is_workload_row);
            enabled.assert_one(local.general.is_ins_row);
            enabled.assert_zero(local.is_first);
        }
        {
            let mut when_transition = builder.when_transition();
            let mut is_ins_row = when_transition.when(local.general.is_ins_row);
            let mut not_first_ins_row = is_ins_row.when_ne(local.is_first, AB::Expr::ONE);
            // ATTENTION: degree of not_first_ins_row is 2
            // Because all the following assert 0, we don't need to check next.enabled.
            // (T6): An Instruction2 row must be followed by a Workload or Disabled row.
            not_first_ins_row.assert_zero(next.prefix.general.is_ins_row);
            // The next row must have idx = 0.
            not_first_ins_row.assert_zero(next.prefix.data.idx);
            // next.result is all 0s
            assert_array_eq(
                &mut not_first_ins_row,
                next.prefix.data.result,
                [AB::Expr::ZERO; EXT_DEG],
            );
        }
    }
}

fn assert_array_eq<AB: AirBuilder, I1: Into<AB::Expr>, I2: Into<AB::Expr>, const N: usize>(
    builder: &mut AB,
    x: [I1; N],
    y: [I2; N],
) {
    for (x, y) in zip_eq(x, y) {
        builder.assert_eq(x, y);
    }
}

fn elem_to_ext<F: Field>(elem: F) -> [F; EXT_DEG] {
    let mut ret = [F::ZERO; EXT_DEG];
    ret[0] = elem;
    ret
}

pub struct FriReducedOpeningStep<F: Field> {
    pub height: usize,
    streams: Arc<Mutex<Streams<F>>>,
}

impl<F: PrimeField32> FriReducedOpeningStep<F> {
    pub fn new(streams: Arc<Mutex<Streams<F>>>) -> Self {
        Self { height: 0, streams }
    }
}

impl<F, CTX> TraceStep<F, CTX> for FriReducedOpeningStep<F>
where
    F: PrimeField32,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, FRI_REDUCED_OPENING.global_opcode().as_usize());
        String::from("FRI_REDUCED_OPENING")
    }

    fn execute(
        &mut self,
        state: VmStateMut<TracingMemory<F>, CTX>,
        instruction: &Instruction<F>,
        trace: &mut [F],
        trace_offset: &mut usize,
        _width: usize,
    ) -> Result<()> {
        let &Instruction {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = instruction;

        let a_ptr_ptr = a.as_canonical_u32();
        let b_ptr_ptr = b.as_canonical_u32();
        let length_ptr = c.as_canonical_u32();
        let alpha_ptr = d.as_canonical_u32();
        let result_ptr = e.as_canonical_u32();
        let hint_id_ptr = f.as_canonical_u32();
        let is_init_ptr = g.as_canonical_u32();

        let timestamp_start = state.memory.timestamp();

        // TODO(ayush): there should be a way to avoid this
        let mut alpha_aux = MemoryReadAuxCols::new(0, LessThanAuxCols::new([F::ZERO; AUX_LEN]));
        let alpha = tracing_read_native(state.memory, alpha_ptr, alpha_aux.as_mut());

        let mut length_aux = MemoryReadAuxCols::new(0, LessThanAuxCols::new([F::ZERO; AUX_LEN]));
        let [length]: [F; 1] = tracing_read_native(state.memory, length_ptr, length_aux.as_mut());

        let mut a_ptr_aux = MemoryReadAuxCols::new(0, LessThanAuxCols::new([F::ZERO; AUX_LEN]));
        let [a_ptr]: [F; 1] = tracing_read_native(state.memory, a_ptr_ptr, a_ptr_aux.as_mut());

        let mut b_ptr_aux = MemoryReadAuxCols::new(0, LessThanAuxCols::new([F::ZERO; AUX_LEN]));
        let [b_ptr]: [F; 1] = tracing_read_native(state.memory, b_ptr_ptr, b_ptr_aux.as_mut());

        let mut is_init_aux = MemoryReadAuxCols::new(0, LessThanAuxCols::new([F::ZERO; AUX_LEN]));
        let [is_init_read]: [F; 1] =
            tracing_read_native(state.memory, is_init_ptr, is_init_aux.as_mut());
        let is_init = is_init_read.as_canonical_u32();

        let [hint_id_f]: [F; 1] = memory_read_native(state.memory.data(), hint_id_ptr);
        let hint_id = hint_id_f.as_canonical_u32() as usize;

        let length = length.as_canonical_u32() as usize;

        let write_a = F::ONE - is_init_read;

        // TODO(ayush): why do we need this?should this be incremented only in tracegen execute?
        // 2 for instruction rows
        self.height += length + 2;

        let data = if is_init == 0 {
            let mut streams = self.streams.lock().unwrap();
            let hint_steam = &mut streams.hint_space[hint_id];
            hint_steam.drain(0..length).collect()
        } else {
            vec![]
        };

        let mut as_and_bs = Vec::with_capacity(length);
        #[allow(clippy::needless_range_loop)]
        for i in 0..length {
            // First read goes to last row
            let start = *trace_offset + (length - i - 1) * OVERALL_WIDTH;
            let cols: &mut WorkloadCols<F> = trace[start..start + WL_WIDTH].borrow_mut();

            let a_ptr_i = (a_ptr + F::from_canonical_usize(i)).as_canonical_u32();
            let [a]: [F; 1] = if is_init == 0 {
                tracing_write_native(state.memory, a_ptr_i, &[data[i]], &mut cols.a_aux);
                [data[i]]
            } else {
                tracing_read_native(state.memory, a_ptr_i, cols.a_aux.as_mut())
            };
            let b_ptr_i = (b_ptr + F::from_canonical_usize(EXT_DEG * i)).as_canonical_u32();
            let b = tracing_read_native::<F, EXT_DEG>(state.memory, b_ptr_i, cols.b_aux.as_mut());

            as_and_bs.push((a, b));
        }

        let mut result = [F::ZERO; EXT_DEG];
        for (i, (a, b)) in as_and_bs.into_iter().rev().enumerate() {
            let start = *trace_offset + i * OVERALL_WIDTH;
            let cols: &mut WorkloadCols<F> = trace[start..start + WL_WIDTH].borrow_mut();

            cols.prefix = PrefixCols {
                general: GeneralCols {
                    is_workload_row: F::ONE,
                    is_ins_row: F::ZERO,
                    timestamp: F::from_canonical_u32(timestamp_start)
                        + F::from_canonical_usize((length - i) * 2),
                },
                a_or_is_first: a,
                data: DataCols {
                    a_ptr: a_ptr + F::from_canonical_usize(length - i),
                    write_a,
                    b_ptr: b_ptr + F::from_canonical_usize((length - i) * EXT_DEG),
                    idx: F::from_canonical_usize(i),
                    result,
                    alpha,
                },
            };
            cols.b = b;

            // result = result * alpha + (b - a)
            result = FieldExtension::add(
                FieldExtension::multiply(result, alpha),
                FieldExtension::subtract(b, elem_to_ext(a)),
            );
        }

        // Instruction1Cols
        {
            let start = *trace_offset + length * OVERALL_WIDTH;
            let cols: &mut Instruction1Cols<F> = trace[start..start + INS_1_WIDTH].borrow_mut();
            *cols = Instruction1Cols {
                prefix: PrefixCols {
                    general: GeneralCols {
                        is_workload_row: F::ZERO,
                        is_ins_row: F::ONE,
                        timestamp: F::from_canonical_u32(timestamp_start),
                    },
                    a_or_is_first: F::ONE,
                    data: DataCols {
                        a_ptr,
                        write_a,
                        b_ptr,
                        idx: F::from_canonical_usize(length),
                        result,
                        alpha,
                    },
                },
                pc: F::from_canonical_u32(*state.pc),
                a_ptr_ptr: a,
                a_ptr_aux,
                b_ptr_ptr: b,
                b_ptr_aux,
                write_a_x_is_first: write_a,
            };
        }

        // Instruction2Cols
        {
            let start = *trace_offset + (length + 1) * OVERALL_WIDTH;
            let cols: &mut Instruction2Cols<F> = trace[start..start + INS_2_WIDTH].borrow_mut();
            cols.general = GeneralCols {
                is_workload_row: F::ZERO,
                is_ins_row: F::ONE,
                timestamp: F::from_canonical_u32(timestamp_start),
            };
            cols.is_first = F::ZERO;
            cols.length_ptr = c;
            cols.length_aux = length_aux;
            cols.alpha_ptr = d;
            cols.alpha_aux = alpha_aux;
            cols.result_ptr = e;
            cols.hint_id_ptr = f;
            cols.is_init_ptr = g;
            cols.is_init_aux = is_init_aux;
            cols.write_a_x_is_first = F::ZERO;

            tracing_write_native(state.memory, result_ptr, &result, &mut cols.result_aux);

            // TODO(ayush): this is a bad hack to make length available to fill_trace_row
            cols.result_aux.base.timestamp_lt_aux.lower_decomp[0] =
                F::from_canonical_u32(length as u32);
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        *trace_offset += (length + 2) * OVERALL_WIDTH;

        Ok(())
    }

    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        let (is_workload_row, is_ins_row) = {
            let cols: &GeneralCols<F> = row_slice[..GENERAL_WIDTH].borrow();
            (cols.is_workload_row.is_one(), cols.is_ins_row.is_one())
        };

        if is_workload_row {
            let cols: &mut WorkloadCols<F> = row_slice[..WL_WIDTH].borrow_mut();

            let timestamp = cols.prefix.general.timestamp.as_canonical_u32();
            mem_helper.fill_from_prev(timestamp + 3, cols.a_aux.as_mut());
            mem_helper.fill_from_prev(timestamp + 4, cols.b_aux.as_mut());
        }

        if is_ins_row {
            let is_ins_1_row = row_slice[GENERAL_WIDTH].is_one();

            if is_ins_1_row {
                let cols: &mut Instruction1Cols<F> = row_slice[..INS_1_WIDTH].borrow_mut();
                let timestamp = cols.prefix.general.timestamp.as_canonical_u32();

                mem_helper.fill_from_prev(timestamp + 2, cols.a_ptr_aux.as_mut());
                mem_helper.fill_from_prev(timestamp + 3, cols.b_ptr_aux.as_mut());
            } else {
                let cols: &mut Instruction2Cols<F> = row_slice[..INS_2_WIDTH].borrow_mut();
                let timestamp = cols.general.timestamp.as_canonical_u32();

                mem_helper.fill_from_prev(timestamp, cols.alpha_aux.as_mut());
                mem_helper.fill_from_prev(timestamp + 1, cols.length_aux.as_mut());
                mem_helper.fill_from_prev(timestamp + 4, cols.is_init_aux.as_mut());

                // TODO(ayush): this is bad
                let length = cols.result_aux.get_base().timestamp_lt_aux.lower_decomp[0];
                mem_helper.fill_from_prev(
                    timestamp + 5 + 2 * length.as_canonical_u32(),
                    cols.result_aux.as_mut(),
                );
            }
        }
    }
}

impl<F> StepExecutorE1<F> for FriReducedOpeningStep<F>
where
    F: PrimeField32,
{
    fn execute_e1<Ctx>(
        &mut self,
        state: &mut VmStateMut<GuestMemory, Ctx>,
        instruction: &Instruction<F>,
    ) -> Result<()>
    where
        Ctx: E1E2ExecutionCtx,
    {
        let &Instruction {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = instruction;

        let a_ptr_ptr = a.as_canonical_u32();
        let b_ptr_ptr = b.as_canonical_u32();
        let length_ptr = c.as_canonical_u32();
        let alpha_ptr = d.as_canonical_u32();
        let result_ptr = e.as_canonical_u32();
        let hint_id_ptr = f.as_canonical_u32();
        let is_init_ptr = g.as_canonical_u32();

        let alpha = memory_read_native(state.memory, alpha_ptr);
        let [length]: [F; 1] = memory_read_native(state.memory, length_ptr);
        let [a_ptr]: [F; 1] = memory_read_native(state.memory, a_ptr_ptr);
        let [b_ptr]: [F; 1] = memory_read_native(state.memory, b_ptr_ptr);
        let [is_init_read]: [F; 1] = memory_read_native(state.memory, is_init_ptr);
        let is_init = is_init_read.as_canonical_u32();

        let [hint_id_f]: [F; 1] = memory_read_native(state.memory, hint_id_ptr);
        let hint_id = hint_id_f.as_canonical_u32() as usize;

        let length = length.as_canonical_u32() as usize;

        let data = if is_init == 0 {
            let mut streams = self.streams.lock().unwrap();
            let hint_steam = &mut streams.hint_space[hint_id];
            hint_steam.drain(0..length).collect()
        } else {
            vec![]
        };

        let mut as_and_bs = Vec::with_capacity(length);
        #[allow(clippy::needless_range_loop)]
        for i in 0..length {
            let a_ptr_i = (a_ptr + F::from_canonical_usize(i)).as_canonical_u32();
            let [a]: [F; 1] = if is_init == 0 {
                memory_write_native(state.memory, a_ptr_i, &[data[i]]);
                [data[i]]
            } else {
                memory_read_native(state.memory, a_ptr_i)
            };
            let b_ptr_i = (b_ptr + F::from_canonical_usize(EXT_DEG * i)).as_canonical_u32();
            let b = memory_read_native::<F, EXT_DEG>(state.memory, b_ptr_i);

            as_and_bs.push((a, b));
        }

        let mut result = [F::ZERO; EXT_DEG];
        for (a, b) in as_and_bs.into_iter().rev() {
            // result = result * alpha + (b - a)
            result = FieldExtension::add(
                FieldExtension::multiply(result, alpha),
                FieldExtension::subtract(b, elem_to_ext(a)),
            );
        }

        // TODO(ayush): why do we need this?should this be incremented only in tracegen execute?
        // 2 for instruction rows
        self.height += length + 2;

        memory_write_native(state.memory, result_ptr, &result);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }

    fn execute_metered(
        &mut self,
        state: &mut VmStateMut<GuestMemory, MeteredCtx>,
        instruction: &Instruction<F>,
        chip_index: usize,
    ) -> Result<()> {
        let &Instruction { c, .. } = instruction;

        let length_ptr = c.as_canonical_u32();
        let [length]: [F; 1] = memory_read_native(state.memory, length_ptr);

        self.execute_e1(state, instruction)?;
        state.ctx.trace_heights[chip_index] += length.as_canonical_u32() + 2;

        Ok(())
    }
}

pub type FriReducedOpeningChip<F> =
    NewVmChipWrapper<F, FriReducedOpeningAir, FriReducedOpeningStep<F>>;
