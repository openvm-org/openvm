use std::sync::Arc;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::Matrix;

use afs_derive::AlignedBorrow;
use afs_primitives::{
    is_zero::{
        columns::{IsZeroCols, IsZeroIoCols},
        IsZeroAir,
    },
    sub_chip::{LocalTraceInstructions, SubAir},
    utils::not,
};
use afs_stark_backend::{Chip, ChipUsageGetter, config::Val, interaction::InteractionBuilder, prover::types::AirProofInput, rap::{AnyRap, BaseAirWithPublicValues}};
use afs_stark_backend::p3_uni_stark::StarkGenericConfig;
use afs_stark_backend::rap::PartitionedBaseAir;
use axvm_instructions::FriFoldOpcode::FRI_FOLD;

use crate::{
    arch::{ExecutionBridge, ExecutionBus, ExecutionState, InstructionExecutor},
    system::{
        memory::{
            MemoryAuxColsFactory,
            MemoryControllerRef, MemoryReadRecord, MemoryWriteRecord, offline_checker::{
                MemoryBaseAuxCols, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols,
            },
        },
        program::{ExecutionError, Instruction, ProgramBus},
    },
};
use crate::system::memory::MemoryAddress;

#[cfg(test)]
mod tests;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FriFoldCols<T> {
    pub enabled: T,
    
    pub pc: T,
    pub start_timestamp: T,

    pub a_pointer: T,
    pub b_pointer: T,
    pub result_pointer: T,
    pub address_space: T,
    pub length_pointer: T,
    pub alpha_pointer: T,
    pub alpha_pow_pointer: T,

    pub a_aux: MemoryReadAuxCols<T, 1>,
    pub b_aux: MemoryReadAuxCols<T, 1>,
    pub result_aux: MemoryWriteAuxCols<T, 1>,
    pub length_aux: MemoryReadAuxCols<T, 1>,
    pub alpha_aux: MemoryReadAuxCols<T, 1>,
    pub alpha_pow_aux: MemoryBaseAuxCols<T>,

    pub a: T,
    pub b: T,
    pub alpha: T,
    pub alpha_pow_original: T,
    pub alpha_pow_current: T,

    pub index: T,
    pub index_is_zero: T,
    pub is_zero_aux: T,
    pub current: T,
}

#[derive(Copy, Clone, Debug)]
pub struct FriFoldAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for FriFoldAir {
    fn width(&self) -> usize {
        FriFoldCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FriFoldAir {}

impl<F: Field> PartitionedBaseAir<F> for FriFoldAir {}

impl<AB: InteractionBuilder> Air<AB> for FriFoldAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &FriFoldCols<AB::Var> = std::borrow::Borrow::borrow(&*local);
        let next = main.row_slice(1);
        let next: &FriFoldCols<AB::Var> = std::borrow::Borrow::borrow(&*next);

        let &FriFoldCols {
            enabled,
            pc,
            start_timestamp,
            a_pointer,
            b_pointer,
            result_pointer,
            address_space,
            length_pointer,
            alpha_pointer,
            alpha_pow_pointer,
            a,
            b,
            alpha,
            alpha_pow_original,
            alpha_pow_current,
            index,
            index_is_zero,
            is_zero_aux,
            current,
            ..
        } = local;
        let a_aux = local.a_aux.clone();
        let b_aux = local.b_aux.clone();
        let result_aux = local.result_aux.clone();
        let length_aux = local.length_aux.clone();
        let alpha_aux = local.alpha_aux.clone();
        let alpha_pow_aux = local.alpha_pow_aux.clone();

        let is_first = index_is_zero;
        let is_last = next.index_is_zero;

        let length = AB::Expr::one() + index;
        let num_initial_accesses = AB::F::two();
        let num_loop_accesses = AB::Expr::two() * length.clone();
        let num_final_accesses = AB::F::two();

        // general constraints

        let mut when_is_not_last = builder.when(not(is_last));

        when_is_not_last.assert_eq(
            next.current,
            (next.alpha_pow_current * (next.b - next.a)) + current,
        );
        when_is_not_last.assert_eq(next.alpha, alpha);
        when_is_not_last.assert_eq(next.alpha_pow_original, alpha_pow_original);
        when_is_not_last.assert_eq(next.alpha_pow_current, alpha_pow_current * alpha);
        when_is_not_last.assert_eq(next.index, index + AB::Expr::one());
        when_is_not_last.assert_eq(next.enabled, enabled);
        
        builder.assert_bool(enabled);

        // first row constraint

        builder
            .when(is_first)
            .assert_eq(alpha_pow_current, alpha_pow_original);
        builder
            .when(is_first)
            .assert_eq(current, alpha_pow_current * (b - a));

        // is zero subair

        SubAir::eval(&IsZeroAir {},
            builder,
            IsZeroIoCols {
                is_zero: index_is_zero,
                x: index,
            },
            is_zero_aux,
        );

        // execution interaction

        let total_accesses = num_loop_accesses.clone() + num_initial_accesses + num_final_accesses;
        self.execution_bridge
            .execute(
                AB::F::from_canonical_usize(FRI_FOLD as usize),
                [
                    a_pointer,
                    b_pointer,
                    result_pointer,
                    address_space,
                    length_pointer,
                    alpha_pointer,
                    alpha_pow_pointer,
                ],
                ExecutionState::new(pc, start_timestamp),
                ExecutionState::<AB::Expr>::new(AB::Expr::one() + pc, total_accesses + start_timestamp - AB::F::one()),
            )
            .eval(builder, enabled * is_last);

        // initial reads

        self.memory_bridge
            .read(MemoryAddress::new(address_space, alpha_pointer), [alpha], start_timestamp, &alpha_aux)
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(MemoryAddress::new(address_space, length_pointer), [length], start_timestamp + AB::F::one(), &length_aux)
            .eval(builder, enabled * is_last);

        // general reads

        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, a_pointer + index),
                [a],
                start_timestamp + num_initial_accesses + (index * AB::F::two()),
                &a_aux,
            )
            .eval(builder, enabled);
        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, b_pointer + index),
                [b],
                start_timestamp + num_initial_accesses + (index * AB::F::two()) + AB::F::one(),
                &b_aux,
            )
            .eval(builder, enabled);

        // final writes

        self.memory_bridge
            .write(
                MemoryAddress::new(address_space, alpha_pow_pointer),
                [alpha * alpha_pow_current],
                start_timestamp + num_initial_accesses + num_loop_accesses.clone(),
                &MemoryWriteAuxCols { base: alpha_pow_aux, prev_data: [alpha_pow_original] },
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .write(
                MemoryAddress::new(address_space, result_pointer),
                [current],
                start_timestamp + num_initial_accesses + num_loop_accesses + AB::F::one(),
                &result_aux,
            )
            .eval(builder, enabled * is_last);
    }
}

pub struct FriFoldRecord<F: Field> {
    pub pc: F,
    pub start_timestamp: F,
    pub instruction: Instruction<F>,
    pub alpha_read: MemoryReadRecord<F, 1>,
    pub length_read: MemoryReadRecord<F, 1>,
    pub a_reads: Vec<MemoryReadRecord<F, 1>>,
    pub b_reads: Vec<MemoryReadRecord<F, 1>>,
    pub alpha_pow_write: MemoryWriteRecord<F, 1>,
    pub result_write: MemoryWriteRecord<F, 1>,
}

pub struct FriFoldChip<F: Field> {
    memory: MemoryControllerRef<F>,
    air: FriFoldAir,
    records: Vec<FriFoldRecord<F>>,
    height: usize,
}

impl<F: PrimeField32> FriFoldChip<F> {
    fn new(
        memory: MemoryControllerRef<F>,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
    ) -> Self {
        let air = FriFoldAir {
            execution_bridge: ExecutionBridge::new(execution_bus, program_bus),
            memory_bridge: memory.borrow().memory_bridge(),
        };
        Self {
            memory,
            records: vec![],
            air,
            height: 0,
        }
    }
}

impl<F: PrimeField32> InstructionExecutor<F> for FriFoldChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            a: a_pointer,
            b: b_pointer,
            c: result_pointer,
            d: address_space,
            e: length_pointer,
            f: alpha_pointer,
            g: alpha_pow_pointer,
            ..
        } = instruction;

        let mut memory = self.memory.borrow_mut();

        let alpha_read = memory.read_cell(address_space, alpha_pointer);
        let length_read = memory.read_cell(address_space, length_pointer);
        let alpha = alpha_read.data[0];
        let alpha_pow_original = memory.unsafe_read_cell(address_space, alpha_pow_pointer);
        let mut alpha_pow = alpha_pow_original;
        let length = length_read.data[0].as_canonical_u32() as usize;

        let mut a_reads = vec![];
        let mut b_reads = vec![];
        let mut result = F::zero();

        for i in 0..length {
            let a_read = memory.read_cell(address_space, a_pointer + F::from_canonical_usize(i));
            let b_read = memory.read_cell(address_space, b_pointer + F::from_canonical_usize(i));
            a_reads.push(a_read);
            b_reads.push(b_read);
            let a = a_read.data[0];
            let b = b_read.data[0];
            result += (b - a) * alpha_pow;
            alpha_pow *= alpha;
        }

        let alpha_pow_write = memory.write_cell(address_space, alpha_pow_pointer, alpha_pow);
        assert_eq!(alpha_pow_write.prev_data[0], alpha_pow_original);
        let result_write = memory.write_cell(address_space, result_pointer, result);

        self.records.push(FriFoldRecord {
            pc: F::from_canonical_u32(from_state.pc),
            start_timestamp: F::from_canonical_u32(from_state.timestamp),
            instruction,
            alpha_read,
            length_read,
            a_reads,
            b_reads,
            alpha_pow_write,
            result_write,
        });
        
        self.height += length;

        Ok(ExecutionState {
            pc: from_state.pc + 1,
            timestamp: result_write.timestamp,
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, FRI_FOLD as usize);
        String::from("FRI_FOLD")
    }
}

impl <F: Field> ChipUsageGetter for FriFoldChip<F> {
    fn air_name(&self) -> String {
        "FriFoldAir".to_string()
    }

    fn current_trace_height(&self) -> usize {
        self.height
    }

    fn trace_width(&self) -> usize {
        FriFoldCols::<F>::width()
    }
}

impl <F: PrimeField32> FriFoldChip<F> {
    fn record_to_rows(
        record: FriFoldRecord<F>,
        aux_cols_factory: MemoryAuxColsFactory<F>,
        slice: &mut [F],
    ) {
        let width = FriFoldCols::<F>::width();

        let Instruction {
            a: a_pointer,
            b: b_pointer,
            c: result_pointer,
            d: address_space,
            e: length_pointer,
            f: alpha_pointer,
            g: alpha_pow_pointer,
            ..
        } = record.instruction;

        let alpha_pow_original = record.alpha_pow_write.prev_data[0];
        let length = record.length_read.data[0].as_canonical_u32() as usize;
        let alpha = record.alpha_read.data[0];

        let mut alpha_pow_current = alpha_pow_original;
        let mut current = F::zero();
        
        let alpha_aux = aux_cols_factory.make_read_aux_cols(record.alpha_read);
        let length_aux = aux_cols_factory.make_read_aux_cols(record.length_read);
        
        let alpha_pow_aux = aux_cols_factory.make_write_aux_cols(record.alpha_pow_write).get_base();
        let result_aux = aux_cols_factory.make_write_aux_cols(record.result_write);

        for i in 0..length {
            let a = record.a_reads[i].data[0];
            let b = record.b_reads[i].data[0];
            current += (b - a) * alpha_pow_current;

            let IsZeroCols {
                io:
                IsZeroIoCols {
                    is_zero: index_is_zero,
                    ..
                },
                inv: is_zero_aux,
            } = IsZeroAir {}.generate_trace_row(F::from_canonical_usize(i));
            
            let cols: &mut FriFoldCols<F> = std::borrow::BorrowMut::borrow_mut(&mut slice[i * width..(i + 1) * width]);
            *cols = FriFoldCols {
                enabled: F::one(),
                pc: record.pc,
                a_pointer,
                b_pointer,
                result_pointer,
                address_space,
                length_pointer,
                alpha_pointer,
                alpha_pow_pointer,
                start_timestamp: record.start_timestamp,
                a_aux: aux_cols_factory.make_read_aux_cols(record.a_reads[i]),
                b_aux: aux_cols_factory.make_read_aux_cols(record.b_reads[i]),
                alpha_aux: alpha_aux.clone(),
                length_aux: length_aux.clone(),
                alpha_pow_aux: alpha_pow_aux.clone(),
                result_aux: result_aux.clone(),
                a,
                b,
                alpha,
                alpha_pow_original,
                alpha_pow_current,
                index: F::from_canonical_usize(i),
                index_is_zero,
                is_zero_aux,
                current,
            };

            alpha_pow_current *= alpha;
        }
    }
    fn blank_row(
        slice: &mut [F],
    ) {
        let cols: &mut FriFoldCols<F> = std::borrow::BorrowMut::borrow_mut(slice);

        let IsZeroCols {
            io:
            IsZeroIoCols {
                is_zero: index_is_zero,
                ..
            },
            inv: is_zero_aux,
        } = IsZeroAir {}.generate_trace_row(F::zero());
        cols.index_is_zero = index_is_zero;
        cols.is_zero_aux = is_zero_aux;
    }
    fn generate_trace(self) -> RowMajorMatrix<F> {
        let mut flat_trace = vec![F::zero(); self.height.next_power_of_two() * self.trace_width()];

        let width = self.trace_width();
        
        let mut index = 0;
        for record in self.records {
            let length = record.a_reads.len();
            Self::record_to_rows(record, self.memory.borrow().aux_cols_factory(), &mut flat_trace[index..index + (length * width)]);
            index += length * width;
        }
        
        while index < flat_trace.len() {
            Self::blank_row(&mut flat_trace[index..index + width]);
            index += width;
        }

        RowMajorMatrix::new(flat_trace, width)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FriFoldChip<Val<SC>> where Val<SC>: PrimeField32 {
    fn air(&self) -> Arc<dyn AnyRap<SC>> {
        Arc::new(self.air)
    }
    fn generate_air_proof_input(self) -> AirProofInput<SC> {
        AirProofInput::simple_no_pis(self.air(), self.generate_trace())
    }
}
