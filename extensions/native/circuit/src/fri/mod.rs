use core::ops::Deref;
use std::{
    borrow::{Borrow, BorrowMut},
    collections::HashMap,
    mem::offset_of,
    sync::{Arc, Mutex},
};

use itertools::{zip_eq, Itertools};
use openvm_circuit::{
    arch::{
        ExecutionBridge, ExecutionBus, ExecutionError, ExecutionState, InstructionExecutor, Streams,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
            MemoryAddress, MemoryAuxColsFactory, MemoryController, OfflineMemory, RecordId,
        },
        program::ProgramBus,
    },
};
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{conversion::AS, FriOpcode::FRI_REDUCED_OPENING};
use openvm_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::types::AirProofInput,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    AirRef, Chip, ChipUsageGetter, Stateful,
};
use serde::{Deserialize, Serialize};
use static_assertions::const_assert_eq;

use crate::field_extension::{FieldExtension, EXT_DEG};

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct WorkloadCols<T> {
    prefix: PrefixCols<T>,

    a_aux: MemoryWriteAuxCols<T, 1>,
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

    /// A trick to reduce 1 degree. When `is_ins_row = 1`, `write_a_x_is_first = write_a * is_first`.
    /// This field must align in both Instruction1Cols and Instruction2Cols.
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
    // is_first = 0 means the second instruction row.
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

    /// A trick to reduce 1 degree. When `is_ins_row = 1`, `write_a_x_is_first = write_a * is_first`.
    /// This field must align in both Instruction1Cols and Instruction2Cols.
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

const fn const_max(a: usize, b: usize) -> usize {
    [a, b][(a < b) as usize]
}
pub const OVERALL_WIDTH: usize = const_max(const_max(WL_WIDTH, INS_1_WIDTH), INS_2_WIDTH);
const_assert_eq!(OVERALL_WIDTH, 27);

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct GeneralCols<T> {
    /// Whether the row is a workload row.
    is_workload_row: T,
    /// Whether the row is an instruction row.
    is_ins_row: T,
    timestamp: T,
}
const GENERAL_WIDTH: usize = GeneralCols::<u8>::width();
const_assert_eq!(GENERAL_WIDTH, 3);

#[repr(C)]
#[derive(Debug, AlignedBorrow)]
struct DataCols<T> {
    a_ptr: T,
    write_a: T,
    b_ptr: T,
    idx: T,
    result: [T; EXT_DEG],
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
    /// WorkloadCols uses this column as `a`. Instruction1Cols uses this column as `is_first` which
    /// indicates whether this is the first row of an instruction row. This is to save a column.
    a_or_is_first: T,
    data: DataCols<T>,
}
const PREFIX_WIDTH: usize = PrefixCols::<u8>::width();
const_assert_eq!(PREFIX_WIDTH, 16);

const INSTRUCTION_READS: usize = 5;

#[derive(Copy, Clone, Debug)]
struct FriReducedOpeningAir {
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
        builder.assert_bool(local.is_ins_row);
        builder.assert_bool(local.is_workload_row);
        // A row can either be an instruction row or a workload row.
        builder.assert_bool(local.is_ins_row + local.is_workload_row);
        {
            // All enabled rows must be before disabled rows.
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
        // write_a itself could be anything on non-workload row, but on workload row, it must be boolean.
        // write_a on last workflow row will be constrained to equal write_a on instruction1 row, implying the latter is boolean.
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
            // The row after a workload row can only be the first instruction row.
            local_non_ins.assert_one(next.a_or_is_first);
        }
        {
            let mut when_first_row = builder.when_first_row();
            let mut when_enabled = when_first_row
                .when(local.prefix.general.is_ins_row + local.prefix.general.is_workload_row);
            // First row must be a workload row.
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
        // These constraints of `is_ins_row` apply to both Instruction1Cols and Instruction2Cols. It's a trick to reduce 1 degree.
        // `write_a` refers to a random field in Instruction2Cols but `write_a_x_is_first` must be 0 because `is_first` is 0.
        is_ins_row.assert_eq(
            local.write_a_x_is_first,
            local.prefix.data.write_a * local.prefix.a_or_is_first,
        );
        is_ins_row.assert_bool(local.write_a_x_is_first);
        let mut is_first_ins = is_ins_row.when(local.prefix.a_or_is_first);
        // ATTENTION: degree of is_first_ins is 2
        is_first_ins.assert_one(next.general.is_ins_row);
        is_first_ins.assert_zero(next.is_first);

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
        {
            let mut last_row = builder.when_last_row();
            let mut enabled =
                last_row.when(local.general.is_ins_row + local.general.is_workload_row);
            // If the last row is enabled, it must be the second row of an instruction row. This
            // is a safeguard for edge cases.
            enabled.assert_one(local.general.is_ins_row);
            enabled.assert_zero(local.is_first);
        }
        {
            let mut when_transition = builder.when_transition();
            let mut is_ins_row = when_transition.when(local.general.is_ins_row);
            let mut not_first_ins_row = is_ins_row.when_ne(local.is_first, AB::Expr::ONE);
            // ATTENTION: degree of not_first_ins_row is 2
            // Because all the following assert 0, we don't need to check next.enabled.
            // The next row must be a workload row.
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

#[derive(Serialize, Deserialize)]
#[serde(bound = "F: Field")]
pub struct FriReducedOpeningRecord<F: Field> {
    pub pc: F,
    pub start_timestamp: F,
    pub instruction: Instruction<F>,
    pub alpha_read: RecordId,
    pub length_read: RecordId,
    pub a_ptr_read: RecordId,
    pub is_init_read: RecordId,
    pub b_ptr_read: RecordId,
    pub a_rws: Vec<RecordId>,
    pub b_reads: Vec<RecordId>,
    pub result_write: RecordId,
}

impl<F: Field> FriReducedOpeningRecord<F> {
    fn get_height(&self) -> usize {
        // 2 for instruction rows
        self.a_rws.len() + 2
    }
}

pub struct FriReducedOpeningChip<F: Field> {
    air: FriReducedOpeningAir,
    records: Vec<FriReducedOpeningRecord<F>>,
    height: usize,
    offline_memory: Arc<Mutex<OfflineMemory<F>>>,
    streams: Arc<Mutex<Streams<F>>>,
    hint_offsets: HashMap<usize, usize>,
}
impl<F: PrimeField32> FriReducedOpeningChip<F> {
    pub fn new(
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        memory_bridge: MemoryBridge,
        offline_memory: Arc<Mutex<OfflineMemory<F>>>,
        streams: Arc<Mutex<Streams<F>>>,
    ) -> Self {
        let air = FriReducedOpeningAir {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge,
        };
        Self {
            records: vec![],
            air,
            height: 0,
            offline_memory,
            streams,
            hint_offsets: HashMap::default(),
        }
    }
}
impl<F: PrimeField32> InstructionExecutor<F> for FriReducedOpeningChip<F> {
    fn execute(
        &mut self,
        memory: &mut MemoryController<F>,
        instruction: &Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let &Instruction {
            a: a_ptr_ptr,
            b: b_ptr_ptr,
            c: length_ptr,
            d: alpha_ptr,
            e: result_ptr,
            f: hint_id_ptr,
            g: is_init_ptr,
            ..
        } = instruction;

        let addr_space = F::from_canonical_u32(AS::Native as u32);
        let alpha_read = memory.read(addr_space, alpha_ptr);
        let length_read = memory.read_cell(addr_space, length_ptr);
        let a_ptr_read = memory.read_cell(addr_space, a_ptr_ptr);
        let b_ptr_read = memory.read_cell(addr_space, b_ptr_ptr);
        let is_init_read = memory.read_cell(addr_space, is_init_ptr);
        let is_init = is_init_read.1.as_canonical_u32();

        let hint_id_f = memory.unsafe_read_cell(addr_space, hint_id_ptr);
        let hint_id = hint_id_f.as_canonical_u32() as usize;
        let hint_offset = self.hint_offsets.get(&hint_id).copied().unwrap_or(0);

        let alpha = alpha_read.1;
        let length = length_read.1.as_canonical_u32() as usize;
        let a_ptr = a_ptr_read.1;
        let b_ptr = b_ptr_read.1;

        let mut a_rws = Vec::with_capacity(length);
        let mut b_reads = Vec::with_capacity(length);
        let mut result = [F::ZERO; EXT_DEG];

        let data = if is_init == 0 {
            self.hint_offsets.insert(hint_id, hint_offset + length);
            let streams = self.streams.lock().unwrap();
            streams.hint_space[hint_id][hint_offset..hint_offset + length].to_vec()
        } else {
            vec![]
        };
        #[allow(clippy::needless_range_loop)]
        for i in 0..length {
            let a_rw = if is_init == 0 {
                let (record_id, _) = memory.write_cell(
                    addr_space,
                    a_ptr + F::from_canonical_usize(i),
                    // Values in the hint space are flattened. We use hint_offset to find the corresponding value.
                    data[i],
                );
                (record_id, data[i])
            } else {
                memory.read_cell(addr_space, a_ptr + F::from_canonical_usize(i))
            };
            let b_read =
                memory.read::<EXT_DEG>(addr_space, b_ptr + F::from_canonical_usize(EXT_DEG * i));
            a_rws.push(a_rw);
            b_reads.push(b_read);
        }

        for (a_rw, b_read) in a_rws.iter().rev().zip_eq(b_reads.iter().rev()) {
            let a = a_rw.1;
            let b = b_read.1;
            // result = result * alpha + (b - a)
            result = FieldExtension::add(
                FieldExtension::multiply(result, alpha),
                FieldExtension::subtract(b, elem_to_ext(a)),
            );
        }

        let (result_write, _) = memory.write(addr_space, result_ptr, result);

        let record = FriReducedOpeningRecord {
            pc: F::from_canonical_u32(from_state.pc),
            start_timestamp: F::from_canonical_u32(from_state.timestamp),
            instruction: instruction.clone(),
            alpha_read: alpha_read.0,
            length_read: length_read.0,
            a_ptr_read: a_ptr_read.0,
            is_init_read: is_init_read.0,
            b_ptr_read: b_ptr_read.0,
            a_rws: a_rws.into_iter().map(|r| r.0).collect(),
            b_reads: b_reads.into_iter().map(|r| r.0).collect(),
            result_write,
        };
        self.height += record.get_height();
        self.records.push(record);

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, FRI_REDUCED_OPENING.global_opcode().as_usize());
        String::from("FRI_REDUCED_OPENING")
    }
}

fn record_to_rows<F: PrimeField32>(
    record: FriReducedOpeningRecord<F>,
    aux_cols_factory: &MemoryAuxColsFactory<F>,
    slice: &mut [F],
    memory: &OfflineMemory<F>,
) {
    let Instruction {
        a: a_ptr_ptr,
        b: b_ptr_ptr,
        c: length_ptr,
        d: alpha_ptr,
        e: result_ptr,
        f: hint_id_ptr,
        g: is_init_ptr,
        ..
    } = record.instruction;

    let length_read = memory.record_by_id(record.length_read);
    let alpha_read = memory.record_by_id(record.alpha_read);
    let a_ptr_read = memory.record_by_id(record.a_ptr_read);
    let b_ptr_read = memory.record_by_id(record.b_ptr_read);
    let is_init_read = memory.record_by_id(record.is_init_read);
    let is_init = is_init_read.data_at(0);
    let write_a = F::ONE - is_init;

    let length = length_read.data_at(0).as_canonical_u32() as usize;
    let alpha: [F; EXT_DEG] = alpha_read.data_slice().try_into().unwrap();
    let a_ptr = a_ptr_read.data_at(0);
    let b_ptr = b_ptr_read.data_at(0);

    let mut result = [F::ZERO; EXT_DEG];

    let alpha_aux = aux_cols_factory.make_read_aux_cols(alpha_read);
    let length_aux = aux_cols_factory.make_read_aux_cols(length_read);
    let a_ptr_aux = aux_cols_factory.make_read_aux_cols(a_ptr_read);
    let b_ptr_aux = aux_cols_factory.make_read_aux_cols(b_ptr_read);
    let is_init_aux = aux_cols_factory.make_read_aux_cols(is_init_read);

    let result_aux = aux_cols_factory.make_write_aux_cols(memory.record_by_id(record.result_write));

    // WorkloadCols
    for (i, (&a_record_id, &b_record_id)) in record
        .a_rws
        .iter()
        .rev()
        .zip_eq(record.b_reads.iter().rev())
        .enumerate()
    {
        let a_rw = memory.record_by_id(a_record_id);
        let b_read = memory.record_by_id(b_record_id);
        let a = a_rw.data_at(0);
        let b: [F; EXT_DEG] = b_read.data_slice().try_into().unwrap();

        let start = i * OVERALL_WIDTH;
        let cols: &mut WorkloadCols<F> = slice[start..start + WL_WIDTH].borrow_mut();
        *cols = WorkloadCols {
            prefix: PrefixCols {
                general: GeneralCols {
                    is_workload_row: F::ONE,
                    is_ins_row: F::ZERO,
                    timestamp: record.start_timestamp + F::from_canonical_usize((length - i) * 2),
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
            },
            // Generate write aux columns no matter `a` is read or written. When `a` is written,
            // `prev_data` is not constrained.
            a_aux: if a_rw.prev_data_slice().is_some() {
                aux_cols_factory.make_write_aux_cols(a_rw)
            } else {
                let read_aux = aux_cols_factory.make_read_aux_cols(a_rw);
                MemoryWriteAuxCols::from_base(read_aux.get_base(), [F::ZERO])
            },
            b,
            b_aux: aux_cols_factory.make_read_aux_cols(b_read),
        };
        // result = result * alpha + (b - a)
        result = FieldExtension::add(
            FieldExtension::multiply(result, alpha),
            FieldExtension::subtract(b, elem_to_ext(a)),
        );
    }
    // Instruction1Cols
    {
        let start = length * OVERALL_WIDTH;
        let cols: &mut Instruction1Cols<F> = slice[start..start + INS_1_WIDTH].borrow_mut();
        *cols = Instruction1Cols {
            prefix: PrefixCols {
                general: GeneralCols {
                    is_workload_row: F::ZERO,
                    is_ins_row: F::ONE,
                    timestamp: record.start_timestamp,
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
            pc: record.pc,
            a_ptr_ptr,
            a_ptr_aux,
            b_ptr_ptr,
            b_ptr_aux,
            write_a_x_is_first: write_a,
        };
    }
    // Instruction2Cols
    {
        let start = (length + 1) * OVERALL_WIDTH;
        let cols: &mut Instruction2Cols<F> = slice[start..start + INS_2_WIDTH].borrow_mut();
        *cols = Instruction2Cols {
            general: GeneralCols {
                is_workload_row: F::ZERO,
                is_ins_row: F::ONE,
                timestamp: record.start_timestamp,
            },
            is_first: F::ZERO,
            length_ptr,
            length_aux,
            alpha_ptr,
            alpha_aux,
            result_ptr,
            result_aux,
            hint_id_ptr,
            is_init_ptr,
            is_init_aux,
            write_a_x_is_first: F::ZERO,
        };
    }
}

impl<F: Field> ChipUsageGetter for FriReducedOpeningChip<F> {
    fn air_name(&self) -> String {
        "FriReducedOpeningAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.height
    }

    fn trace_width(&self) -> usize {
        OVERALL_WIDTH
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FriReducedOpeningChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air)
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        let height = next_power_of_two_or_zero(self.height);
        let mut flat_trace = Val::<SC>::zero_vec(OVERALL_WIDTH * height);
        let chunked_trace = {
            let sizes: Vec<_> = self
                .records
                .par_iter()
                .map(|record| OVERALL_WIDTH * record.get_height())
                .collect();
            variable_chunks_mut(&mut flat_trace, &sizes)
        };

        let memory = self.offline_memory.lock().unwrap();
        let aux_cols_factory = memory.aux_cols_factory();

        self.records
            .into_par_iter()
            .zip_eq(chunked_trace.into_par_iter())
            .for_each(|(record, slice)| {
                record_to_rows(record, &aux_cols_factory, slice, &memory);
            });

        let matrix = RowMajorMatrix::new(flat_trace, OVERALL_WIDTH);
        AirProofInput::simple_no_pis(matrix)
    }
}

impl<F: PrimeField32> Stateful<Vec<u8>> for FriReducedOpeningChip<F> {
    fn load_state(&mut self, state: Vec<u8>) {
        self.records = bitcode::deserialize(&state).unwrap();
        self.height = self.records.iter().map(|record| record.get_height()).sum();
    }

    fn store_state(&self) -> Vec<u8> {
        bitcode::serialize(&self.records).unwrap()
    }
}

fn variable_chunks_mut<'a, T>(mut slice: &'a mut [T], sizes: &[usize]) -> Vec<&'a mut [T]> {
    let mut result = Vec::with_capacity(sizes.len());
    for &size in sizes {
        // split_at_mut guarantees disjoint slices
        let (left, right) = slice.split_at_mut(size);
        result.push(left);
        slice = right; // move forward for the next chunk
    }
    result
}
