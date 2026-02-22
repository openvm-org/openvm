use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        BasicAdapterInterface, ExecutionBridge, ExecutionState, MinimalInstruction, VmAdapterAir,
    },
    system::{
        memory::{
            offline_checker::{MemoryBridge, MemoryWriteAuxCols, MemoryWriteAuxRecord},
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::tracing_write_native,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, NATIVE_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_air::BaseAir, p3_field::PrimeCharacteristicRing,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

use crate::{
    setup::EmptyRecord,
    utils::{memory_op_chunk, split_memory_ops, DIGEST_MEMORY_OPS, MEMORY_OP_SIZE},
};

// ========================= AIR ==============================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralSetupAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub write_aux: [MemoryWriteAuxCols<T, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralSetupAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    native_start_ptr: u32,
}

impl<F> BaseAir<F> for DeferralSetupAdapterAir {
    fn width(&self) -> usize {
        DeferralSetupAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for DeferralSetupAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 0, 1, 0, DIGEST_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &DeferralSetupAdapterCols<_> = local.borrow();

        // The SETUP opcode should always write to the same location in the native
        // address space. Field native_start_ptr is a config value.
        let address_space = AB::Expr::from_u32(NATIVE_AS);
        let pointer = AB::Expr::from_u32(self.native_start_ptr);
        let [write_data] = ctx.writes;
        let write_chunks = split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(write_data);

        for (chunk_idx, (data, aux)) in write_chunks.into_iter().zip(&cols.write_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        address_space.clone(),
                        pointer.clone() + AB::Expr::from_usize(chunk_idx * MEMORY_OP_SIZE),
                    ),
                    data,
                    cols.from_state.timestamp + AB::Expr::from_usize(chunk_idx),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    pointer,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    address_space,
                    AB::Expr::ZERO,
                ],
                cols.from_state,
                AB::Expr::from_usize(DIGEST_MEMORY_OPS),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &DeferralSetupAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

// ========================= EXECUTION + TRACEGEN ==============================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct DeferralSetupAdapterRecord<F> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub write_aux: [MemoryWriteAuxRecord<F, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(derive_new::new, Clone, Copy)]
pub struct DeferralSetupAdapterExecutor {
    pub(in crate::setup) native_start_ptr: u32,
}

#[derive(derive_new::new)]
pub struct DeferralSetupAdapterFiller;

impl<F: PrimeField32> AdapterTraceExecutor<F> for DeferralSetupAdapterExecutor {
    const WIDTH: usize = DeferralSetupAdapterCols::<u8>::width();
    type ReadData = EmptyRecord;
    type WriteData = [F; DIGEST_SIZE];
    type RecordMut<'a> = &'a mut DeferralSetupAdapterRecord<F>;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        _memory: &mut TracingMemory,
        _instruction: &Instruction<F>,
        _record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        EmptyRecord
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { a, d, .. } = instruction;
        debug_assert_eq!(a.as_canonical_u32(), self.native_start_ptr);
        debug_assert_eq!(d.as_canonical_u32(), NATIVE_AS);
        for chunk_idx in 0..DIGEST_MEMORY_OPS {
            tracing_write_native(
                memory,
                self.native_start_ptr + (chunk_idx * MEMORY_OP_SIZE) as u32,
                memory_op_chunk(&data, chunk_idx),
                &mut record.write_aux[chunk_idx].prev_timestamp,
                &mut record.write_aux[chunk_idx].prev_data,
            );
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for DeferralSetupAdapterFiller {
    const WIDTH: usize = DeferralSetupAdapterCols::<u8>::width();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: caller ensures `adapter_row` contains a valid record representation
        // that was previously written by the executor
        let record: &DeferralSetupAdapterRecord<F> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut DeferralSetupAdapterCols<F> = adapter_row.borrow_mut();

        // Writing in reverse order to avoid overwriting the record
        for chunk_idx in (0..DIGEST_MEMORY_OPS).rev() {
            adapter_row.write_aux[chunk_idx].set_prev_data(record.write_aux[chunk_idx].prev_data);
            mem_helper.fill(
                record.write_aux[chunk_idx].prev_timestamp,
                record.from_timestamp + (chunk_idx as u32),
                adapter_row.write_aux[chunk_idx].as_mut(),
            );
        }
        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
