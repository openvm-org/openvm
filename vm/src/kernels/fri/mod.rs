use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    sync::Arc,
};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::{
    is_zero::{IsZeroIo, IsZeroSubAir},
    utils::{assert_array_eq, not},
    SubAir, TraceSubRowGenerator,
};
use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    interaction::InteractionBuilder,
    prover::types::AirProofInput,
    rap::{AnyRap, BaseAirWithPublicValues, PartitionedBaseAir},
    Chip, ChipUsageGetter,
};
use axvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, FriOpcode::FRI_MAT_OPENING,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor},
    kernels::field_extension::FieldExtension,
    system::{
        memory::{
            offline_checker::{
                MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols,
            },
            MemoryAddress, MemoryAuxColsFactory, MemoryControllerRef, MemoryReadRecord,
            MemoryWriteRecord,
        },
        program::{ExecutionError, ProgramBus},
    },
};

#[cfg(test)]
mod tests;

pub const EXT_DEG: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FriMatOpeningCols<T> {
    pub enabled: T,

    pub pc: T,
    pub start_timestamp: T,

    pub a_ptr_ptr: T,
    pub b_ptr_ptr: T,
    pub result_ptr: T,
    pub addr_space: T,
    pub length_ptr: T,
    pub alpha_ptr: T,
    pub alpha_pow_ptr: T,

    pub a_ptr_aux: MemoryReadAuxCols<T, 1>,
    pub b_ptr_aux: MemoryReadAuxCols<T, 1>,
    pub a_aux: MemoryReadAuxCols<T, 1>,
    pub b_aux: MemoryReadAuxCols<T, EXT_DEG>,
    pub result_aux: MemoryWriteAuxCols<T, EXT_DEG>,
    pub length_aux: MemoryReadAuxCols<T, 1>,
    pub alpha_aux: MemoryReadAuxCols<T, EXT_DEG>,
    pub alpha_pow_aux: MemoryBaseAuxCols<T>,

    pub a_ptr: T,
    pub b_ptr: T,
    pub a: T,
    pub b: [T; EXT_DEG],
    pub alpha: [T; EXT_DEG],
    pub alpha_pow_original: [T; EXT_DEG],
    pub alpha_pow_current: [T; EXT_DEG],

    pub idx: T,
    pub idx_is_zero: T,
    pub is_zero_aux: T,
    pub current: [T; EXT_DEG],
}

#[derive(Copy, Clone, Debug)]
pub struct FriMatOpeningAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    offset: usize,
}

impl<F: Field> BaseAir<F> for FriMatOpeningAir {
    fn width(&self) -> usize {
        FriMatOpeningCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FriMatOpeningAir {}
impl<F: Field> PartitionedBaseAir<F> for FriMatOpeningAir {}

impl<AB: InteractionBuilder> Air<AB> for FriMatOpeningAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &FriMatOpeningCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &FriMatOpeningCols<AB::Var> = (*next).borrow();

        let &FriMatOpeningCols {
            enabled,
            pc,
            start_timestamp,
            a_ptr_ptr,
            b_ptr_ptr,
            result_ptr,
            addr_space,
            length_ptr,
            alpha_ptr,
            alpha_pow_ptr,
            a_ptr,
            b_ptr,
            a,
            b,
            alpha,
            alpha_pow_original,
            alpha_pow_current,
            idx,
            idx_is_zero,
            is_zero_aux,
            current,
            a_ptr_aux,
            b_ptr_aux,
            a_aux,
            b_aux,
            result_aux,
            length_aux,
            alpha_aux,
            alpha_pow_aux,
        } = local;

        let is_first = idx_is_zero;
        let is_last = next.idx_is_zero;

        builder.assert_bool(enabled);
        // transition constraints
        let mut when_is_not_last = builder.when(not(is_last));

        let next_alpha_pow_times_b = FieldExtension::multiply(next.alpha_pow_current, next.b);
        for i in 0..EXT_DEG {
            when_is_not_last.assert_eq(
                next.current[i],
                next_alpha_pow_times_b[i].clone() - (next.alpha_pow_current[i] * next.a)
                    + current[i],
            );
        }

        assert_array_eq(&mut when_is_not_last, next.alpha, alpha);
        assert_array_eq(
            &mut when_is_not_last,
            next.alpha_pow_original,
            alpha_pow_original,
        );
        assert_array_eq(
            &mut when_is_not_last,
            next.alpha_pow_current,
            FieldExtension::multiply(alpha, alpha_pow_current),
        );
        when_is_not_last.assert_eq(next.idx, idx + AB::Expr::ONE);
        when_is_not_last.assert_eq(next.enabled, enabled);
        when_is_not_last.assert_eq(next.start_timestamp, start_timestamp);

        // first row constraint
        assert_array_eq(
            &mut builder.when(is_first),
            alpha_pow_current,
            alpha_pow_original,
        );

        let alpha_pow_times_b = FieldExtension::multiply(alpha_pow_current, b);
        for i in 0..EXT_DEG {
            builder.when(is_first).assert_eq(
                current[i],
                alpha_pow_times_b[i].clone() - (alpha_pow_current[i] * a),
            );
        }

        // is zero constraint
        let is_zero_io = IsZeroIo::new(idx.into(), idx_is_zero.into(), AB::Expr::ONE);
        IsZeroSubAir.eval(builder, (is_zero_io, is_zero_aux));

        // length will only be used on the last row, so it equals 1 + idx
        let length = AB::Expr::ONE + idx;
        let num_initial_accesses = AB::F::from_canonical_usize(4);
        let num_loop_accesses = AB::Expr::TWO * length.clone();
        let num_final_accesses = AB::F::TWO;

        // execution interaction
        let total_accesses = num_loop_accesses.clone() + num_initial_accesses + num_final_accesses;
        self.execution_bridge
            .execute(
                AB::F::from_canonical_usize((FRI_MAT_OPENING as usize) + self.offset),
                [
                    a_ptr_ptr,
                    b_ptr_ptr,
                    result_ptr,
                    addr_space,
                    length_ptr,
                    alpha_ptr,
                    alpha_pow_ptr,
                ],
                ExecutionState::new(pc, start_timestamp),
                ExecutionState::<AB::Expr>::new(
                    AB::Expr::from_canonical_u32(DEFAULT_PC_STEP) + pc,
                    start_timestamp + total_accesses,
                ),
            )
            .eval(builder, enabled * is_last);

        // initial reads
        self.memory_bridge
            .read(
                MemoryAddress::new(addr_space, alpha_ptr),
                alpha,
                start_timestamp,
                &alpha_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(addr_space, length_ptr),
                [length],
                start_timestamp + AB::F::ONE,
                &length_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(addr_space, a_ptr_ptr),
                [a_ptr],
                start_timestamp + AB::F::TWO,
                &a_ptr_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(addr_space, b_ptr_ptr),
                [b_ptr],
                start_timestamp + AB::F::from_canonical_usize(3),
                &b_ptr_aux,
            )
            .eval(builder, enabled * is_last);

        // general reads
        let timestamp = start_timestamp + num_initial_accesses + (idx * AB::F::TWO);
        self.memory_bridge
            .read(
                MemoryAddress::new(addr_space, a_ptr + idx),
                [a],
                timestamp.clone(),
                &a_aux,
            )
            .eval(builder, enabled);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    addr_space,
                    b_ptr + (idx * AB::F::from_canonical_usize(EXT_DEG)),
                ),
                b,
                timestamp + AB::F::ONE,
                &b_aux,
            )
            .eval(builder, enabled);

        // final writes
        let timestamp = start_timestamp + num_initial_accesses + num_loop_accesses.clone();
        self.memory_bridge
            .write(
                MemoryAddress::new(addr_space, alpha_pow_ptr),
                FieldExtension::multiply(alpha, alpha_pow_current),
                timestamp.clone(),
                &MemoryWriteAuxCols {
                    base: alpha_pow_aux,
                    prev_data: alpha_pow_original,
                },
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .write(
                MemoryAddress::new(addr_space, result_ptr),
                current,
                timestamp + AB::F::ONE,
                &result_aux,
            )
            .eval(builder, enabled * is_last);
    }
}

pub struct FriMatOpeningRecord<F: Field> {
    pub pc: F,
    pub start_timestamp: F,
    pub instruction: Instruction<F>,
    pub alpha_read: MemoryReadRecord<F, EXT_DEG>,
    pub length_read: MemoryReadRecord<F, 1>,
    pub a_ptr_read: MemoryReadRecord<F, 1>,
    pub b_ptr_read: MemoryReadRecord<F, 1>,
    pub a_reads: Vec<MemoryReadRecord<F, 1>>,
    pub b_reads: Vec<MemoryReadRecord<F, EXT_DEG>>,
    pub alpha_pow_write: MemoryWriteRecord<F, EXT_DEG>,
    pub result_write: MemoryWriteRecord<F, EXT_DEG>,
}

pub struct FriMatOpeningChip<F: Field> {
    memory: MemoryControllerRef<F>,
    air: FriMatOpeningAir,
    records: Vec<FriMatOpeningRecord<F>>,
    height: usize,
}

impl<F: PrimeField32> FriMatOpeningChip<F> {
    pub fn new(
        memory: MemoryControllerRef<F>,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        offset: usize,
    ) -> Self {
        let air = FriMatOpeningAir {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge: RefCell::borrow(&memory).memory_bridge(),
            offset,
        };
        Self {
            memory,
            records: vec![],
            air,
            height: 0,
        }
    }
}

fn elem_to_ext<F: Field>(elem: F) -> [F; EXT_DEG] {
    let mut ret = [F::ZERO; EXT_DEG];
    ret[0] = elem;
    ret
}

impl<F: PrimeField32> InstructionExecutor<F> for FriMatOpeningChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            a: a_ptr_ptr,
            b: b_ptr_ptr,
            c: result_ptr,
            d: addr_space,
            e: length_ptr,
            f: alpha_ptr,
            g: alpha_pow_ptr,
            ..
        } = instruction;

        let mut memory = RefCell::borrow_mut(&self.memory);

        let alpha_read = memory.read(addr_space, alpha_ptr);
        let length_read = memory.read_cell(addr_space, length_ptr);
        let a_ptr_read = memory.read_cell(addr_space, a_ptr_ptr);
        let b_ptr_read = memory.read_cell(addr_space, b_ptr_ptr);

        let alpha = alpha_read.data;
        let alpha_pow_original = from_fn(|i| {
            memory.unsafe_read_cell(addr_space, alpha_pow_ptr + F::from_canonical_usize(i))
        });
        let mut alpha_pow = alpha_pow_original;
        let length = length_read.data[0].as_canonical_u32() as usize;
        let a_ptr = a_ptr_read.data[0];
        let b_ptr = b_ptr_read.data[0];

        let mut a_reads = Vec::with_capacity(length);
        let mut b_reads = Vec::with_capacity(length);
        let mut result = [F::ZERO; EXT_DEG];

        for i in 0..length {
            let a_read = memory.read_cell(addr_space, a_ptr + F::from_canonical_usize(i));
            let b_read = memory.read(addr_space, b_ptr + F::from_canonical_usize(4 * i));
            a_reads.push(a_read);
            b_reads.push(b_read);
            let a = a_read.data[0];
            let b = b_read.data;
            result = FieldExtension::add(
                result,
                FieldExtension::multiply(FieldExtension::subtract(b, elem_to_ext(a)), alpha_pow),
            );
            alpha_pow = FieldExtension::multiply(alpha, alpha_pow);
        }

        let alpha_pow_write = memory.write(addr_space, alpha_pow_ptr, alpha_pow);
        debug_assert_eq!(alpha_pow_write.prev_data, alpha_pow_original);
        let result_write = memory.write(addr_space, result_ptr, result);

        self.records.push(FriMatOpeningRecord {
            pc: F::from_canonical_u32(from_state.pc),
            start_timestamp: F::from_canonical_u32(from_state.timestamp),
            instruction,
            alpha_read,
            length_read,
            a_ptr_read,
            b_ptr_read,
            a_reads,
            b_reads,
            alpha_pow_write,
            result_write,
        });

        self.height += length;

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: memory.timestamp(),
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, (FRI_MAT_OPENING as usize) + self.air.offset);
        String::from("FRI_FOLD")
    }
}

impl<F: Field> ChipUsageGetter for FriMatOpeningChip<F> {
    fn air_name(&self) -> String {
        "FriMatOpeningAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.height
    }

    fn trace_width(&self) -> usize {
        FriMatOpeningCols::<F>::width()
    }
}

impl<F: PrimeField32> FriMatOpeningChip<F> {
    fn record_to_rows(
        record: FriMatOpeningRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
    ) {
        let width = FriMatOpeningCols::<F>::width();

        let Instruction {
            a: a_ptr_ptr,
            b: b_ptr_ptr,
            c: result_ptr,
            d: addr_space,
            e: length_ptr,
            f: alpha_ptr,
            g: alpha_pow_ptr,
            ..
        } = record.instruction;

        let alpha_pow_original = record.alpha_pow_write.prev_data;
        let length = record.length_read.data[0].as_canonical_u32() as usize;
        let alpha = record.alpha_read.data;
        let a_ptr = record.a_ptr_read.data[0];
        let b_ptr = record.b_ptr_read.data[0];

        let mut alpha_pow_current = alpha_pow_original;
        let mut current = [F::ZERO; EXT_DEG];

        let alpha_aux = aux_cols_factory.make_read_aux_cols(record.alpha_read);
        let length_aux = aux_cols_factory.make_read_aux_cols(record.length_read);
        let a_ptr_aux = aux_cols_factory.make_read_aux_cols(record.a_ptr_read);
        let b_ptr_aux = aux_cols_factory.make_read_aux_cols(record.b_ptr_read);

        let alpha_pow_aux = aux_cols_factory
            .make_write_aux_cols(record.alpha_pow_write)
            .get_base();
        let result_aux = aux_cols_factory.make_write_aux_cols(record.result_write);

        for i in 0..length {
            let a = record.a_reads[i].data[0];
            let b = record.b_reads[i].data;
            current = FieldExtension::add(
                current,
                FieldExtension::multiply(
                    FieldExtension::subtract(b, elem_to_ext(a)),
                    alpha_pow_current,
                ),
            );

            let mut idx_is_zero = F::ZERO;
            let mut is_zero_aux = F::ZERO;

            let idx = F::from_canonical_usize(i);
            IsZeroSubAir.generate_subrow(idx, (&mut is_zero_aux, &mut idx_is_zero));

            let cols: &mut FriMatOpeningCols<F> = slice[i * width..(i + 1) * width].borrow_mut();
            *cols = FriMatOpeningCols {
                enabled: F::ONE,
                pc: record.pc,
                a_ptr_ptr,
                b_ptr_ptr,
                result_ptr,
                addr_space,
                length_ptr,
                alpha_ptr,
                alpha_pow_ptr,
                start_timestamp: record.start_timestamp,
                a_ptr_aux,
                b_ptr_aux,
                a_aux: aux_cols_factory.make_read_aux_cols(record.a_reads[i]),
                b_aux: aux_cols_factory.make_read_aux_cols(record.b_reads[i]),
                alpha_aux,
                length_aux,
                alpha_pow_aux,
                result_aux,
                a_ptr,
                b_ptr,
                a,
                b,
                alpha,
                alpha_pow_original,
                alpha_pow_current,
                idx,
                idx_is_zero,
                is_zero_aux,
                current,
            };

            alpha_pow_current = FieldExtension::multiply(alpha, alpha_pow_current);
        }
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let mut flat_trace = vec![F::ZERO; self.height.next_power_of_two() * self.trace_width()];
        let width = self.trace_width();
        let aux_cols_factory = RefCell::borrow(&self.memory).aux_cols_factory();

        let mut idx = 0;
        for record in self.records {
            let length = record.a_reads.len();
            Self::record_to_rows(
                record,
                &aux_cols_factory,
                &mut flat_trace[idx..idx + (length * width)],
            );
            idx += length * width;
        }
        // In padding rows, need idx_is_zero = 1 so IsZero constraints pass, and also because next.idx_is_zero is used
        // to determine the last row per instruction, so the last non-padding row needs next.idx_is_zero = 1
        flat_trace[self.height * width..]
            .par_chunks_mut(width)
            .for_each(|row| {
                let row: &mut FriMatOpeningCols<F> = row.borrow_mut();
                row.idx_is_zero = F::ONE;
            });

        RowMajorMatrix::new(flat_trace, width)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FriMatOpeningChip<Val<SC>>
where
    Val<SC>: PrimeField32,
{
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        AirProofInput::simple_no_pis(self.air(), self.generate_trace())
    }
}
