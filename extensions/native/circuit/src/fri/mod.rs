use core::ops::Deref;
use std::{
    borrow::{Borrow, BorrowMut},
    mem::offset_of,
};

use itertools::zip_eq;
use openvm_circuit::{
    arch::*,
    system::{
        memory::{
            offline_checker::{
                MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
                MemoryWriteAuxRecord,
            },
            online::{GuestMemory, TracingMemory},
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{memory_read_native, tracing_read_native, tracing_write_native},
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FriOpcode::FRI_REDUCED_OPENING};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::{BinomialExtensionField, BinomiallyExtendable},
        ExtensionField, Field, FieldAlgebra, FieldExtensionAlgebra, PackedField, PackedValue,
        PrimeField32,
    },
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use static_assertions::const_assert_eq;

use crate::{
    field_extension::{FieldExtension, EXT_DEG},
    transmute_array_to_ext, transmute_ext_to_array,
    utils::const_max,
};

type EF<F> = BinomialExtensionField<F, EXT_DEG>;
type ExtPacked<F> = <EF<F> as ExtensionField<F>>::ExtensionPacking;

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

#[derive(Copy, Clone, Debug)]
pub struct FriReducedOpeningMetadata {
    length: usize,
    is_init: bool,
}

impl MultiRowMetadata for FriReducedOpeningMetadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        // Allocates `length` workload rows + 1 Instruction1 row + 1 Instruction2 row
        self.length + 2
    }
}

type FriReducedOpeningLayout = MultiRowLayout<FriReducedOpeningMetadata>;

// Header of record that is common for all trace rows for an instruction
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FriReducedOpeningHeaderRecord {
    pub length: u32,
    pub is_init: bool,
}

// Part of record that is common for all trace rows for an instruction
// NOTE: Order for fields is important here to prevent overwriting.
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FriReducedOpeningCommonRecord<F> {
    pub timestamp: u32,

    pub a_ptr: u32,

    pub b_ptr: u32,

    pub alpha: [F; EXT_DEG],

    pub from_pc: u32,

    pub a_ptr_ptr: F,
    pub a_ptr_aux: MemoryReadAuxRecord,

    pub b_ptr_ptr: F,
    pub b_ptr_aux: MemoryReadAuxRecord,

    pub length_ptr: F,
    pub length_aux: MemoryReadAuxRecord,

    pub alpha_ptr: F,
    pub alpha_aux: MemoryReadAuxRecord,

    pub result_ptr: F,
    pub result_aux: MemoryWriteAuxRecord<F, EXT_DEG>,

    pub hint_id_ptr: F,

    pub is_init_ptr: F,
    pub is_init_aux: MemoryReadAuxRecord,
}

// Part of record for each workload row - stores a,b values directly
// NOTE: Order for fields is important here to prevent overwriting.
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct FriReducedOpeningWorkloadRowRecord<F> {
    pub a: F,
    pub a_aux: MemoryReadAuxRecord,
    // Store b values directly for vectorized computation
    pub b: [F; EXT_DEG],
    pub b_aux: MemoryReadAuxRecord,
}

// NOTE: Order for fields is important here to prevent overwriting.
#[derive(Debug)]
pub struct FriReducedOpeningRecordMut<'a, F> {
    pub header: &'a mut FriReducedOpeningHeaderRecord,
    pub workload: &'a mut [FriReducedOpeningWorkloadRowRecord<F>],
    // if is_init this will be an empty slice, otherwise it will be the previous data of writing
    // `a`s
    pub a_write_prev_data: &'a mut [F],
    pub common: &'a mut FriReducedOpeningCommonRecord<F>,
}

impl<'a, F> CustomBorrow<'a, FriReducedOpeningRecordMut<'a, F>, FriReducedOpeningLayout>
    for [u8]
{
    fn custom_borrow(
        &'a mut self,
        layout: FriReducedOpeningLayout,
    ) -> FriReducedOpeningRecordMut<'a, F> {
        let (header_buf, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<FriReducedOpeningHeaderRecord>()) };
        let header: &mut FriReducedOpeningHeaderRecord = header_buf.borrow_mut();

        let workload_size =
            layout.metadata.length * size_of::<FriReducedOpeningWorkloadRowRecord<F>>();

        let (workload_buf, rest) = unsafe { rest.split_at_mut_unchecked(workload_size) };
        let a_prev_size = if layout.metadata.is_init {
            0
        } else {
            layout.metadata.length * size_of::<F>()
        };

        let (a_prev_buf, common_buf) = unsafe { rest.split_at_mut_unchecked(a_prev_size) };

        let (_, a_prev_records, _) = unsafe { a_prev_buf.align_to_mut::<F>() };
        let (_, workload_records, _) =
            unsafe { workload_buf.align_to_mut::<FriReducedOpeningWorkloadRowRecord<F>>() };

        let common: &mut FriReducedOpeningCommonRecord<F> = common_buf.borrow_mut();

        FriReducedOpeningRecordMut {
            header,
            workload: &mut workload_records[..layout.metadata.length],
            a_write_prev_data: &mut a_prev_records[..],
            common,
        }
    }

    unsafe fn extract_layout(&self) -> FriReducedOpeningLayout {
        let header: &FriReducedOpeningHeaderRecord = self.borrow();
        FriReducedOpeningLayout::new(FriReducedOpeningMetadata {
            length: header.length as usize,
            is_init: header.is_init,
        })
    }
}

impl<F> SizedRecord<FriReducedOpeningLayout> for FriReducedOpeningRecordMut<'_, F> {
    fn size(layout: &FriReducedOpeningLayout) -> usize {
        let mut total_len = size_of::<FriReducedOpeningHeaderRecord>();
        total_len += layout.metadata.length * size_of::<FriReducedOpeningWorkloadRowRecord<F>>();
        if !layout.metadata.is_init {
            total_len += layout.metadata.length * size_of::<F>();
        }
        total_len += size_of::<FriReducedOpeningCommonRecord<F>>();
        total_len
    }

    fn alignment(_layout: &FriReducedOpeningLayout) -> usize {
        align_of::<FriReducedOpeningHeaderRecord>()
    }
}

#[derive(derive_new::new, Copy, Clone)]
pub struct FriReducedOpeningExecutor;

#[derive(derive_new::new)]
pub struct FriReducedOpeningFiller;

pub type FriReducedOpeningChip<F> = VmChipWrapper<F, FriReducedOpeningFiller>;

impl Default for FriReducedOpeningExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for FriReducedOpeningExecutor
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    for<'buf> RA: RecordArena<'buf, FriReducedOpeningLayout, FriReducedOpeningRecordMut<'buf, F>>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, FRI_REDUCED_OPENING.global_opcode().as_usize());
        String::from("FRI_REDUCED_OPENING")
    }

    fn execute(
        &mut self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
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

        let timestamp_start = state.memory.timestamp;

        // Read length from memory to allocate record
        let length_ptr = c.as_canonical_u32();
        let [length]: [F; 1] = memory_read_native(&state.memory.data, length_ptr);
        let length = length.as_canonical_u32();
        let is_init_ptr = g.as_canonical_u32();
        let [is_init]: [F; 1] = memory_read_native(&state.memory.data, is_init_ptr);
        let is_init = is_init != F::ZERO;

        let metadata = FriReducedOpeningMetadata {
            length: length as usize,
            is_init,
        };
        let record = state.ctx.alloc(MultiRowLayout::new(metadata));

        record.common.from_pc = *state.pc;
        record.common.timestamp = timestamp_start;

        let alpha_ptr = d.as_canonical_u32();
        let alpha = tracing_read_native(
            state.memory,
            alpha_ptr,
            &mut record.common.alpha_aux.prev_timestamp,
        );
        record.common.alpha_ptr = d;
        record.common.alpha = alpha;

        tracing_read_native::<F, 1>(
            state.memory,
            length_ptr,
            &mut record.common.length_aux.prev_timestamp,
        );
        record.common.length_ptr = c;
        record.header.length = length;

        let a_ptr_ptr = a.as_canonical_u32();
        let [a_ptr]: [F; 1] = tracing_read_native(
            state.memory,
            a_ptr_ptr,
            &mut record.common.a_ptr_aux.prev_timestamp,
        );
        record.common.a_ptr_ptr = a;
        record.common.a_ptr = a_ptr.as_canonical_u32();

        let b_ptr_ptr = b.as_canonical_u32();
        let [b_ptr]: [F; 1] = tracing_read_native(
            state.memory,
            b_ptr_ptr,
            &mut record.common.b_ptr_aux.prev_timestamp,
        );
        record.common.b_ptr_ptr = b;
        record.common.b_ptr = b_ptr.as_canonical_u32();

        tracing_read_native::<F, 1>(
            state.memory,
            is_init_ptr,
            &mut record.common.is_init_aux.prev_timestamp,
        );
        record.common.is_init_ptr = g;
        record.header.is_init = is_init;

        let hint_id_ptr = f.as_canonical_u32();
        let [hint_id]: [F; 1] = memory_read_native(state.memory.data(), hint_id_ptr);
        let hint_id = hint_id.as_canonical_u32() as usize;
        record.common.hint_id_ptr = f;

        let length = length as usize;

        let data = if !is_init {
            let hint_steam = &mut state.streams.hint_space[hint_id];
            hint_steam.drain(0..length).collect()
        } else {
            vec![]
        };

        let mut as_and_bs: Vec<(F, EF<F>)> = Vec::with_capacity(length);
        #[allow(clippy::needless_range_loop)]
        for i in 0..length {
            let workload_row = &mut record.workload[length - i - 1];

            let a_ptr_i = record.common.a_ptr + i as u32;
            let [a]: [F; 1] = if !is_init {
                let mut prev = [F::ZERO; 1];
                tracing_write_native(
                    state.memory,
                    a_ptr_i,
                    [data[i]],
                    &mut workload_row.a_aux.prev_timestamp,
                    &mut prev,
                );
                record.a_write_prev_data[length - i - 1] = prev[0];
                [data[i]]
            } else {
                tracing_read_native(
                    state.memory,
                    a_ptr_i,
                    &mut workload_row.a_aux.prev_timestamp,
                )
            };
            let b_ptr_i = record.common.b_ptr + (EXT_DEG * i) as u32;
            let b = tracing_read_native::<F, EXT_DEG>(
                state.memory,
                b_ptr_i,
                &mut workload_row.b_aux.prev_timestamp,
            );

            // Store a and b values directly in record for fill_trace
            workload_row.a = a;
            workload_row.b = b;

            // Convert b to extension field for vectorized computation
            let b_ext = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&b);
            as_and_bs.push((a, b_ext));
        }

        // Use vectorized polynomial evaluation like execute_e12
        let alpha_ext = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&alpha);
        let result_ext = compute_polynomial_evaluation(&as_and_bs, alpha_ext);
        let result = transmute_ext_to_array::<F, EF<F>, EXT_DEG>(&result_ext);

        let result_ptr = e.as_canonical_u32();
        tracing_write_native(
            state.memory,
            result_ptr,
            result,
            &mut record.common.result_aux.prev_timestamp,
            &mut record.common.result_aux.prev_data,
        );
        record.common.result_ptr = e;

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32 + BinomiallyExtendable<EXT_DEG>> TraceFiller<F> for FriReducedOpeningFiller {
    fn fill_trace(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        trace: &mut RowMajorMatrix<F>,
        rows_used: usize,
    ) {
        if rows_used == 0 {
            return;
        }
        debug_assert_eq!(trace.width, OVERALL_WIDTH);

        let mut remaining_trace = &mut trace.values[..OVERALL_WIDTH * rows_used];
        let mut chunks = Vec::with_capacity(rows_used);
        while !remaining_trace.is_empty() {
            let header: &FriReducedOpeningHeaderRecord =
                unsafe { get_record_from_slice(&mut remaining_trace, ()) };
            let num_rows = header.length as usize + 2;
            let chunk_size = OVERALL_WIDTH * num_rows;
            let (chunk, rest) = remaining_trace.split_at_mut(chunk_size);
            chunks.push((chunk, header.is_init));
            remaining_trace = rest;
        }

        chunks.into_par_iter().for_each(|(mut chunk, is_init)| {
            let num_rows = chunk.len() / OVERALL_WIDTH;
            let metadata = FriReducedOpeningMetadata {
                length: num_rows - 2,
                is_init,
            };
            let record: FriReducedOpeningRecordMut<F> =
                unsafe { get_record_from_slice(&mut chunk, MultiRowLayout::new(metadata)) };

            let timestamp = record.common.timestamp;
            let length = record.header.length as usize;
            let alpha = record.common.alpha;
            let is_init = record.header.is_init;
            let write_a = F::from_bool(!is_init);

            let a_ptr = record.common.a_ptr;
            let b_ptr = record.common.b_ptr;

            let (workload_chunk, rest) = chunk.split_at_mut(length * OVERALL_WIDTH);
            let (ins1_chunk, ins2_chunk) = rest.split_at_mut(OVERALL_WIDTH);

            // Pre-compute results array by computing rolling polynomial result
            let mut results: Vec<[F; EXT_DEG]> =
                std::iter::once([F::ZERO; EXT_DEG])
                    .chain(record.workload.iter().scan(
                        [F::ZERO; EXT_DEG],
                        |result, workload_row| {
                            let a = workload_row.a;
                            let b = workload_row.b;

                            *result = FieldExtension::add(
                                FieldExtension::multiply(*result, alpha),
                                FieldExtension::subtract(b, elem_to_ext(a)),
                            );
                            Some(*result)
                        },
                    ))
                    .collect();

            {
                // ins2 row
                let cols: &mut Instruction2Cols<F> = ins2_chunk[..INS_2_WIDTH].borrow_mut();

                cols.write_a_x_is_first = F::ZERO;

                mem_helper.fill(
                    record.common.is_init_aux.prev_timestamp,
                    timestamp + 4,
                    cols.is_init_aux.as_mut(),
                );
                cols.is_init_ptr = record.common.is_init_ptr;

                cols.hint_id_ptr = record.common.hint_id_ptr;

                cols.result_aux
                    .set_prev_data(record.common.result_aux.prev_data);
                mem_helper.fill(
                    record.common.result_aux.prev_timestamp,
                    timestamp + 5 + 2 * length as u32,
                    cols.result_aux.as_mut(),
                );
                cols.result_ptr = record.common.result_ptr;

                mem_helper.fill(
                    record.common.alpha_aux.prev_timestamp,
                    timestamp,
                    cols.alpha_aux.as_mut(),
                );
                cols.alpha_ptr = record.common.alpha_ptr;

                mem_helper.fill(
                    record.common.length_aux.prev_timestamp,
                    timestamp + 1,
                    cols.length_aux.as_mut(),
                );
                cols.length_ptr = record.common.length_ptr;

                cols.is_first = F::ZERO;

                cols.general.timestamp = F::from_canonical_u32(timestamp);
                cols.general.is_ins_row = F::ONE;
                cols.general.is_workload_row = F::ZERO;

                ins2_chunk[INS_2_WIDTH..OVERALL_WIDTH].fill(F::ZERO);
            }

            {
                // ins 1 row
                let cols: &mut Instruction1Cols<F> = ins1_chunk[..INS_1_WIDTH].borrow_mut();

                cols.write_a_x_is_first = write_a;

                mem_helper.fill(
                    record.common.b_ptr_aux.prev_timestamp,
                    timestamp + 3,
                    cols.b_ptr_aux.as_mut(),
                );
                cols.b_ptr_ptr = record.common.b_ptr_ptr;

                mem_helper.fill(
                    record.common.a_ptr_aux.prev_timestamp,
                    timestamp + 2,
                    cols.a_ptr_aux.as_mut(),
                );
                cols.a_ptr_ptr = record.common.a_ptr_ptr;

                cols.pc = F::from_canonical_u32(record.common.from_pc);

                cols.prefix.data.alpha = alpha;
                cols.prefix.data.result = results.pop().unwrap();
                cols.prefix.data.idx = F::from_canonical_usize(length);
                cols.prefix.data.b_ptr = F::from_canonical_u32(b_ptr);
                cols.prefix.data.write_a = write_a;
                cols.prefix.data.a_ptr = F::from_canonical_u32(a_ptr);

                cols.prefix.a_or_is_first = F::ONE;

                cols.prefix.general.timestamp = F::from_canonical_u32(timestamp);
                cols.prefix.general.is_ins_row = F::ONE;
                cols.prefix.general.is_workload_row = F::ZERO;
                ins1_chunk[INS_1_WIDTH..OVERALL_WIDTH].fill(F::ZERO);
            }

            // To fill the WorkloadRows we do 2 passes:
            // - First, a serial pass to fill some of the records into the trace
            // - Then, a parallel pass to fill the rest of the records into the trace
            // Note, the first pass is done to avoid overwriting the records

            // Copy of `a_write_prev_data` to avoid overwriting it and to use it in the parallel
            // pass
            let a_prev_data = if !is_init {
                let mut tmp = Vec::with_capacity(length);
                tmp.extend_from_slice(record.a_write_prev_data);
                tmp
            } else {
                vec![]
            };

            for (i, (workload_row, row_chunk)) in record
                .workload
                .iter()
                .zip(workload_chunk.chunks_exact_mut(OVERALL_WIDTH))
                .enumerate()
                .rev()
            {
                let cols: &mut WorkloadCols<F> = row_chunk[..WL_WIDTH].borrow_mut();

                let timestamp = timestamp + ((length - i) * 2) as u32;

                // fill in reverse order
                mem_helper.fill(
                    workload_row.b_aux.prev_timestamp,
                    timestamp + 4,
                    cols.b_aux.as_mut(),
                );

                // Store the actual b values directly
                cols.b = workload_row.b;

                mem_helper.fill(
                    workload_row.a_aux.prev_timestamp,
                    timestamp + 3,
                    cols.a_aux.as_mut(),
                );
                cols.prefix.a_or_is_first = workload_row.a;
            }

            workload_chunk
                .par_chunks_exact_mut(OVERALL_WIDTH)
                .enumerate()
                .for_each(|(i, row_chunk)| {
                    let cols: &mut WorkloadCols<F> = row_chunk[..WL_WIDTH].borrow_mut();
                    let timestamp = timestamp + ((length - i) * 2) as u32;
                    if is_init {
                        cols.a_aux.set_prev_data([F::ZERO; 1]);
                    } else {
                        cols.a_aux.set_prev_data([a_prev_data[i]]);
                    }

                    // DataCols
                    cols.prefix.data.a_ptr = F::from_canonical_u32(a_ptr + (length - i) as u32);
                    cols.prefix.data.write_a = write_a;
                    cols.prefix.data.b_ptr =
                        F::from_canonical_u32(b_ptr + ((length - i) * EXT_DEG) as u32);
                    cols.prefix.data.idx = F::from_canonical_usize(i);
                    cols.prefix.data.result = results[i];
                    cols.prefix.data.alpha = alpha;

                    // GeneralCols
                    cols.prefix.general.is_workload_row = F::ONE;
                    cols.prefix.general.is_ins_row = F::ZERO;

                    // WorkloadCols
                    cols.prefix.general.timestamp = F::from_canonical_u32(timestamp);

                    row_chunk[WL_WIDTH..OVERALL_WIDTH].fill(F::ZERO);
                });
        });
    }
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct FriReducedOpeningPreCompute {
    a_ptr_ptr: u32,
    b_ptr_ptr: u32,
    length_ptr: u32,
    alpha_ptr: u32,
    result_ptr: u32,
    hint_id_ptr: u32,
    is_init_ptr: u32,
}

impl FriReducedOpeningExecutor {
    #[inline(always)]
    fn pre_compute_impl<F: PrimeField32>(
        &self,
        _pc: u32,
        inst: &Instruction<F>,
        data: &mut FriReducedOpeningPreCompute,
    ) -> Result<(), StaticProgramError> {
        let &Instruction {
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        let a_ptr_ptr = a.as_canonical_u32();
        let b_ptr_ptr = b.as_canonical_u32();
        let length_ptr = c.as_canonical_u32();
        let alpha_ptr = d.as_canonical_u32();
        let result_ptr = e.as_canonical_u32();
        let hint_id_ptr = f.as_canonical_u32();
        let is_init_ptr = g.as_canonical_u32();

        *data = FriReducedOpeningPreCompute {
            a_ptr_ptr,
            b_ptr_ptr,
            length_ptr,
            alpha_ptr,
            result_ptr,
            hint_id_ptr,
            is_init_ptr,
        };

        Ok(())
    }
}

impl<F> Executor<F> for FriReducedOpeningExecutor
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        size_of::<FriReducedOpeningPreCompute>()
    }

    #[inline(always)]
    fn pre_compute<Ctx: E1ExecutionCtx>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut FriReducedOpeningPreCompute = data.borrow_mut();

        self.pre_compute_impl(pc, inst, pre_compute)?;

        let fn_ptr = execute_e1_impl;
        Ok(fn_ptr)
    }
}

impl<F> MeteredExecutor<F> for FriReducedOpeningExecutor
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        size_of::<E2PreCompute<FriReducedOpeningPreCompute>>()
    }

    #[inline(always)]
    fn metered_pre_compute<Ctx: E2ExecutionCtx>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        let pre_compute: &mut E2PreCompute<FriReducedOpeningPreCompute> = data.borrow_mut();
        pre_compute.chip_idx = chip_idx as u32;

        self.pre_compute_impl(pc, inst, &mut pre_compute.data)?;

        let fn_ptr = execute_e2_impl;
        Ok(fn_ptr)
    }
}

unsafe fn execute_e1_impl<F, CTX>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: E1ExecutionCtx,
{
    let pre_compute: &FriReducedOpeningPreCompute = pre_compute.borrow();
    execute_e12_impl(pre_compute, vm_state);
}

unsafe fn execute_e2_impl<F, CTX>(
    pre_compute: &[u8],
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: E2ExecutionCtx,
{
    let pre_compute: &E2PreCompute<FriReducedOpeningPreCompute> = pre_compute.borrow();
    let height = execute_e12_impl(&pre_compute.data, vm_state);
    vm_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[inline(always)]
unsafe fn execute_e12_impl<F, CTX>(
    pre_compute: &FriReducedOpeningPreCompute,
    vm_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    CTX: E1ExecutionCtx,
{
    let alpha = vm_state.vm_read(AS::Native as u32, pre_compute.alpha_ptr);
    let alpha = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&alpha);

    let [length]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.length_ptr);
    let length = length.as_canonical_u32() as usize;

    let [a_ptr]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.a_ptr_ptr);
    let [b_ptr]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.b_ptr_ptr);

    let [is_init_read]: [F; 1] = vm_state.vm_read(AS::Native as u32, pre_compute.is_init_ptr);
    let is_init = is_init_read.as_canonical_u32();

    let [hint_id_f]: [F; 1] = vm_state.host_read(AS::Native as u32, pre_compute.hint_id_ptr);
    let hint_id = hint_id_f.as_canonical_u32() as usize;

    let data = if is_init == 0 {
        let hint_steam = &mut vm_state.streams.hint_space[hint_id];
        hint_steam.drain(0..length).collect()
    } else {
        vec![]
    };

    let mut as_and_bs: Vec<(F, EF<F>)> = Vec::with_capacity(length);
    #[allow(clippy::needless_range_loop)]
    for i in 0..length {
        let a_ptr_i = (a_ptr + F::from_canonical_usize(i)).as_canonical_u32();
        let [a]: [F; 1] = if is_init == 0 {
            vm_state.vm_write(AS::Native as u32, a_ptr_i, &[data[i]]);
            [data[i]]
        } else {
            vm_state.vm_read(AS::Native as u32, a_ptr_i)
        };
        let b_ptr_i = (b_ptr + F::from_canonical_usize(EXT_DEG * i)).as_canonical_u32();
        let b = vm_state.vm_read(AS::Native as u32, b_ptr_i);
        let b = transmute_array_to_ext::<F, EF<F>, EXT_DEG>(&b);

        as_and_bs.push((a, b));
    }

    // Use vectorized polynomial evaluation
    let result = compute_polynomial_evaluation(&as_and_bs, alpha);

    let result = transmute_ext_to_array::<F, EF<F>, EXT_DEG>(&result);
    vm_state.vm_write(AS::Native as u32, pre_compute.result_ptr, &result);

    vm_state.pc = vm_state.pc.wrapping_add(DEFAULT_PC_STEP);
    vm_state.instret += 1;

    length as u32 + 2
}

fn compute_polynomial_evaluation<F>(as_and_bs: &[(F, EF<F>)], alpha: EF<F>) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    F::Packing: PackedField<Scalar = F>,
    EF<F>: ExtensionField<F>,
{
    match F::Packing::WIDTH {
        4 => packed_polynomial_evaluation::<F, 4>(as_and_bs, alpha),
        8 => packed_polynomial_evaluation::<F, 8>(as_and_bs, alpha),
        16 => packed_polynomial_evaluation::<F, 16>(as_and_bs, alpha),
        _ => scalar_polynomial_evaluation(as_and_bs, alpha),
    }
}

#[inline]
fn scalar_polynomial_evaluation<F>(as_and_bs: &[(F, EF<F>)], alpha: EF<F>) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
{
    let mut result = EF::<F>::ZERO;
    for &(a, b) in as_and_bs.iter().rev() {
        result = result * alpha + (b - EF::<F>::from_base(a));
    }
    result
}

#[inline]
fn packed_polynomial_evaluation<F, const WIDTH: usize>(
    as_and_bs: &[(F, EF<F>)],
    alpha: EF<F>,
) -> EF<F>
where
    F: PrimeField32 + BinomiallyExtendable<EXT_DEG>,
    F::Packing: PackedField<Scalar = F>,
    EF<F>: ExtensionField<F>,
{
    let mut alpha_powers = [EF::<F>::ONE; WIDTH];
    for i in 1..WIDTH {
        alpha_powers[i] = alpha_powers[i - 1] * alpha;
    }

    let mut alpha_powers_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
        F::Packing::from_fn(|lane| alpha_powers[lane].as_base_slice()[coeff_idx])
    });

    let alpha_width = alpha_powers[WIDTH - 1] * alpha;
    let alpha_width_packed = ExtPacked::<F>::from_f(alpha_width);

    let len = as_and_bs.len();
    let mut result = EF::<F>::ZERO;

    for batch_start in (0..len).step_by(WIDTH) {
        let batch_end = (batch_start + WIDTH).min(len);
        let batch_size = batch_end - batch_start;

        let mut batch_coeffs = [[F::ZERO; WIDTH]; EXT_DEG];
        for j in 0..batch_size {
            let (a, b) = as_and_bs[batch_start + j];
            let diff = b - EF::<F>::from_base(a);

            let diff_slice = diff.as_base_slice();
            for coeff in 0..EXT_DEG {
                batch_coeffs[coeff][j] = diff_slice[coeff];
            }
        }

        let coeffs_packed = ExtPacked::<F>::from_base_fn(|coeff_idx| {
            F::Packing::from_fn(|lane| batch_coeffs[coeff_idx][lane])
        });

        let batch_result_packed = coeffs_packed * alpha_powers_packed;

        for lane in 0..WIDTH {
            let lane_result = EF::<F>::from_base_fn(|coeff_idx| {
                batch_result_packed.as_base_slice()[coeff_idx].as_slice()[lane]
            });
            result += lane_result;
        }

        alpha_powers_packed *= alpha_width_packed;
    }

    result
}
