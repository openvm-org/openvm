use std::{borrow::Borrow, cell::RefCell, sync::Arc};

use ax_circuit_derive::AlignedBorrow;
use ax_circuit_primitives::{
    is_zero::{IsZeroIo, IsZeroSubAir},
    utils::not,
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
    instruction::Instruction, program::DEFAULT_PC_STEP, FriFoldOpcode::FRI_FOLD,
};
use itertools::zip_eq;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

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
pub struct FriFoldCols<T> {
    pub enabled: T,

    pub pc: T,
    pub start_timestamp: T,

    pub a_pointer_pointer: T,
    pub b_pointer_pointer: T,
    pub result_pointer: T,
    pub address_space: T,
    pub length_pointer: T,
    pub alpha_pointer: T,
    pub alpha_pow_pointer: T,

    pub a_pointer_aux: MemoryReadAuxCols<T, 1>,
    pub b_pointer_aux: MemoryReadAuxCols<T, 1>,
    pub a_aux: MemoryReadAuxCols<T, 1>,
    pub b_aux: MemoryReadAuxCols<T, EXT_DEG>,
    pub result_aux: MemoryWriteAuxCols<T, EXT_DEG>,
    pub length_aux: MemoryReadAuxCols<T, 1>,
    pub alpha_aux: MemoryReadAuxCols<T, EXT_DEG>,
    pub alpha_pow_aux: MemoryBaseAuxCols<T>,

    pub a_pointer: T,
    pub b_pointer: T,
    pub a: T,
    pub b: [T; EXT_DEG],
    pub alpha: [T; EXT_DEG],
    pub alpha_pow_original: [T; EXT_DEG],
    pub alpha_pow_current: [T; EXT_DEG],

    pub index: T,
    pub index_is_zero: T,
    pub is_zero_aux: T,
    pub current: [T; EXT_DEG],
}

#[derive(Copy, Clone, Debug)]
pub struct FriFoldAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    offset: usize,
}

impl<F: Field> BaseAir<F> for FriFoldAir {
    fn width(&self) -> usize {
        FriFoldCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for FriFoldAir {}
impl<F: Field> PartitionedBaseAir<F> for FriFoldAir {}

fn assert_eq_ext<AB: AirBuilder, I1: Into<AB::Expr>, I2: Into<AB::Expr>>(
    builder: &mut AB,
    x: [I1; EXT_DEG],
    y: [I2; EXT_DEG],
) {
    for (x, y) in zip_eq(x, y) {
        builder.assert_eq(x, y);
    }
}

impl<AB: InteractionBuilder> Air<AB> for FriFoldAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &FriFoldCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &FriFoldCols<AB::Var> = (*next).borrow();

        let &FriFoldCols {
            enabled,
            pc,
            start_timestamp,
            a_pointer_pointer,
            b_pointer_pointer,
            result_pointer,
            address_space,
            length_pointer,
            alpha_pointer,
            alpha_pow_pointer,
            a_pointer,
            b_pointer,
            a,
            b,
            alpha,
            alpha_pow_original,
            alpha_pow_current,
            index,
            index_is_zero,
            is_zero_aux,
            current,
            a_pointer_aux,
            b_pointer_aux,
            a_aux,
            b_aux,
            result_aux,
            length_aux,
            alpha_aux,
            alpha_pow_aux,
        } = local;

        let is_first = index_is_zero;
        let is_last = next.index_is_zero;

        let length = AB::Expr::one() + index;
        let num_initial_accesses = AB::F::from_canonical_usize(4);
        let num_loop_accesses = AB::Expr::two() * length.clone();
        let num_final_accesses = AB::F::two();

        // general constraints
        let mut when_is_not_last = builder.when(not(is_last));

        let next_alpha_pow_times_b = FieldExtension::multiply(next.alpha_pow_current, next.b);
        for i in 0..EXT_DEG {
            when_is_not_last.assert_eq(
                next.current[i],
                next_alpha_pow_times_b[i].clone() - (next.alpha_pow_current[i] * next.a)
                    + current[i],
            );
        }

        assert_eq_ext(&mut when_is_not_last, next.alpha, alpha);
        assert_eq_ext(
            &mut when_is_not_last,
            next.alpha_pow_original,
            alpha_pow_original,
        );
        assert_eq_ext(
            &mut when_is_not_last,
            next.alpha_pow_current,
            FieldExtension::multiply(alpha, alpha_pow_current),
        );
        when_is_not_last.assert_eq(next.index, index + AB::Expr::one());
        when_is_not_last.assert_eq(next.enabled, enabled);

        builder.assert_bool(enabled);

        // first row constraint
        assert_eq_ext(
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
        let is_zero_io = IsZeroIo::new(index.into(), index_is_zero.into(), enabled.into());
        IsZeroSubAir.eval(builder, (is_zero_io, is_zero_aux));

        // execution interaction

        let total_accesses = num_loop_accesses.clone() + num_initial_accesses + num_final_accesses;
        self.execution_bridge
            .execute(
                AB::F::from_canonical_usize((FRI_FOLD as usize) + self.offset),
                [
                    a_pointer_pointer,
                    b_pointer_pointer,
                    result_pointer,
                    address_space,
                    length_pointer,
                    alpha_pointer,
                    alpha_pow_pointer,
                ],
                ExecutionState::new(pc, start_timestamp),
                ExecutionState::<AB::Expr>::new(
                    AB::Expr::from_canonical_u32(DEFAULT_PC_STEP) + pc,
                    total_accesses + start_timestamp - AB::F::one(),
                ),
            )
            .eval(builder, enabled * is_last);

        // initial reads

        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, alpha_pointer),
                alpha,
                start_timestamp,
                &alpha_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, length_pointer),
                [length],
                start_timestamp + AB::F::one(),
                &length_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, a_pointer_pointer),
                [a_pointer],
                start_timestamp + AB::F::two(),
                &a_pointer_aux,
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .read(
                MemoryAddress::new(address_space, b_pointer_pointer),
                [b_pointer],
                start_timestamp + AB::F::from_canonical_usize(3),
                &b_pointer_aux,
            )
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
                MemoryAddress::new(
                    address_space,
                    b_pointer + (index * AB::F::from_canonical_usize(4)),
                ),
                b,
                start_timestamp + num_initial_accesses + (index * AB::F::two()) + AB::F::one(),
                &b_aux,
            )
            .eval(builder, enabled);

        // final writes

        self.memory_bridge
            .write(
                MemoryAddress::new(address_space, alpha_pow_pointer),
                FieldExtension::multiply(alpha, alpha_pow_current),
                start_timestamp + num_initial_accesses + num_loop_accesses.clone(),
                &MemoryWriteAuxCols {
                    base: alpha_pow_aux,
                    prev_data: alpha_pow_original,
                },
            )
            .eval(builder, enabled * is_last);
        self.memory_bridge
            .write(
                MemoryAddress::new(address_space, result_pointer),
                current,
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
    pub alpha_read: MemoryReadRecord<F, EXT_DEG>,
    pub length_read: MemoryReadRecord<F, 1>,
    pub a_pointer_read: MemoryReadRecord<F, 1>,
    pub b_pointer_read: MemoryReadRecord<F, 1>,
    pub a_reads: Vec<MemoryReadRecord<F, 1>>,
    pub b_reads: Vec<MemoryReadRecord<F, EXT_DEG>>,
    pub alpha_pow_write: MemoryWriteRecord<F, EXT_DEG>,
    pub result_write: MemoryWriteRecord<F, EXT_DEG>,
}

pub struct FriFoldChip<F: Field> {
    memory: MemoryControllerRef<F>,
    air: FriFoldAir,
    records: Vec<FriFoldRecord<F>>,
    height: usize,
}

impl<F: PrimeField32> FriFoldChip<F> {
    #[allow(dead_code)]
    pub(crate) fn new(
        memory: MemoryControllerRef<F>,
        execution_bus: ExecutionBus,
        program_bus: ProgramBus,
        offset: usize,
    ) -> Self {
        let air = FriFoldAir {
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
    let mut ret = [F::zero(); EXT_DEG];
    ret[0] = elem;
    ret
}

impl<F: PrimeField32> InstructionExecutor<F> for FriFoldChip<F> {
    fn execute(
        &mut self,
        instruction: Instruction<F>,
        from_state: ExecutionState<u32>,
    ) -> Result<ExecutionState<u32>, ExecutionError> {
        let Instruction {
            a: a_pointer_pointer,
            b: b_pointer_pointer,
            c: result_pointer,
            d: address_space,
            e: length_pointer,
            f: alpha_pointer,
            g: alpha_pow_pointer,
            ..
        } = instruction;

        let mut memory = self.memory.borrow_mut();

        let alpha_read = memory.read(address_space, alpha_pointer);
        let length_read = memory.read_cell(address_space, length_pointer);
        let a_pointer_read = memory.read_cell(address_space, a_pointer_pointer);
        let b_pointer_read = memory.read_cell(address_space, b_pointer_pointer);

        let alpha = alpha_read.data;
        let alpha_pow_original = std::array::from_fn(|i| {
            memory.unsafe_read_cell(
                address_space,
                alpha_pow_pointer + F::from_canonical_usize(i),
            )
        });
        let mut alpha_pow = alpha_pow_original;
        let length = length_read.data[0].as_canonical_u32() as usize;
        let a_pointer = a_pointer_read.data[0];
        let b_pointer = b_pointer_read.data[0];

        let mut a_reads = vec![];
        let mut b_reads = vec![];
        let mut result = [F::zero(); EXT_DEG];

        for i in 0..length {
            let a_read = memory.read_cell(address_space, a_pointer + F::from_canonical_usize(i));
            let b_read = memory.read(address_space, b_pointer + F::from_canonical_usize(4 * i));
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

        let alpha_pow_write = memory.write(address_space, alpha_pow_pointer, alpha_pow);
        assert_eq!(alpha_pow_write.prev_data, alpha_pow_original);
        let result_write = memory.write(address_space, result_pointer, result);

        self.records.push(FriFoldRecord {
            pc: F::from_canonical_u32(from_state.pc),
            start_timestamp: F::from_canonical_u32(from_state.timestamp),
            instruction,
            alpha_read,
            length_read,
            a_pointer_read,
            b_pointer_read,
            a_reads,
            b_reads,
            alpha_pow_write,
            result_write,
        });

        self.height += length;

        Ok(ExecutionState {
            pc: from_state.pc + DEFAULT_PC_STEP,
            timestamp: result_write.timestamp,
        })
    }

    fn get_opcode_name(&self, opcode: usize) -> String {
        assert_eq!(opcode, (FRI_FOLD as usize) + self.air.offset);
        String::from("FRI_FOLD")
    }
}

impl<F: Field> ChipUsageGetter for FriFoldChip<F> {
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

impl<F: PrimeField32> FriFoldChip<F> {
    fn record_to_rows(
        record: FriFoldRecord<F>,
        aux_cols_factory: &MemoryAuxColsFactory<F>,
        slice: &mut [F],
    ) {
        let width = FriFoldCols::<F>::width();

        let Instruction {
            a: a_pointer_pointer,
            b: b_pointer_pointer,
            c: result_pointer,
            d: address_space,
            e: length_pointer,
            f: alpha_pointer,
            g: alpha_pow_pointer,
            ..
        } = record.instruction;

        let alpha_pow_original = record.alpha_pow_write.prev_data;
        let length = record.length_read.data[0].as_canonical_u32() as usize;
        let alpha = record.alpha_read.data;
        let a_pointer = record.a_pointer_read.data[0];
        let b_pointer = record.b_pointer_read.data[0];

        let mut alpha_pow_current = alpha_pow_original;
        let mut current = [F::zero(); EXT_DEG];

        let alpha_aux = aux_cols_factory.make_read_aux_cols(record.alpha_read);
        let length_aux = aux_cols_factory.make_read_aux_cols(record.length_read);
        let a_pointer_aux = aux_cols_factory.make_read_aux_cols(record.a_pointer_read);
        let b_pointer_aux = aux_cols_factory.make_read_aux_cols(record.b_pointer_read);

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

            let mut index_is_zero = F::zero();
            let mut is_zero_aux = F::zero();

            let index = F::from_canonical_usize(i);
            IsZeroSubAir {}.generate_subrow(index, (&mut is_zero_aux, &mut index_is_zero));

            let cols: &mut FriFoldCols<F> =
                std::borrow::BorrowMut::borrow_mut(&mut slice[i * width..(i + 1) * width]);
            *cols = FriFoldCols {
                enabled: F::one(),
                pc: record.pc,
                a_pointer_pointer,
                b_pointer_pointer,
                result_pointer,
                address_space,
                length_pointer,
                alpha_pointer,
                alpha_pow_pointer,
                start_timestamp: record.start_timestamp,
                a_pointer_aux,
                b_pointer_aux,
                a_aux: aux_cols_factory.make_read_aux_cols(record.a_reads[i]),
                b_aux: aux_cols_factory.make_read_aux_cols(record.b_reads[i]),
                alpha_aux,
                length_aux,
                alpha_pow_aux,
                result_aux,
                a_pointer,
                b_pointer,
                a,
                b,
                alpha,
                alpha_pow_original,
                alpha_pow_current,
                index,
                index_is_zero,
                is_zero_aux,
                current,
            };

            alpha_pow_current = FieldExtension::multiply(alpha, alpha_pow_current);
        }
    }

    fn generate_trace(self) -> RowMajorMatrix<F> {
        let mut flat_trace = vec![F::zero(); self.height.next_power_of_two() * self.trace_width()];
        let width = self.trace_width();
        let aux_cols_factory = RefCell::borrow(&self.memory).aux_cols_factory();

        let mut index = 0;
        for record in self.records {
            let length = record.a_reads.len();
            Self::record_to_rows(
                record,
                &aux_cols_factory,
                &mut flat_trace[index..index + (length * width)],
            );
            index += length * width;
        }

        RowMajorMatrix::new(flat_trace, width)
    }
}

impl<SC: StarkGenericConfig> Chip<SC> for FriFoldChip<Val<SC>>
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
