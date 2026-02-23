use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterTraceExecutor, AdapterTraceFiller, EmptyAdapterCoreLayout,
        ExecutionError, PreflightExecutor, RecordArena, TraceFiller, VmStateMut,
    },
    system::{
        memory::{
            offline_checker::MemoryWriteAuxRecord, online::TracingMemory, MemoryAuxColsFactory,
        },
        native_adapter::util::tracing_write_native,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, NATIVE_AS};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

use crate::{
    setup::{DeferralSetupAdapterCols, DeferralSetupCoreCols},
    utils::{memory_op_chunk, DIGEST_MEMORY_OPS, MEMORY_OP_SIZE},
};

// ========================= CORE ==============================

#[derive(AlignedBytesBorrow)]
pub struct EmptyRecord;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralSetupCoreExecutor<F, A> {
    pub(in crate::setup) adapter: A,
    pub(in crate::setup) expected_def_vks_commit: [F; DIGEST_SIZE],
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralSetupCoreFiller<A> {
    adapter: A,
}

impl<F, A, RA> PreflightExecutor<F, RA> for DeferralSetupCoreExecutor<F, A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceExecutor<F, ReadData = EmptyRecord, WriteData = [F; DIGEST_SIZE]>,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut EmptyRecord),
    >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::SETUP)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, _core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        self.adapter.write(
            state.memory,
            instruction,
            self.expected_def_vks_commit,
            &mut adapter_record,
        );
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for DeferralSetupCoreFiller<A>
where
    F: PrimeField32,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // DeferralSetupCoreCols::width() elements
        let (adapter_row, core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);
        let core_row: &mut DeferralSetupCoreCols<F> = core_row.borrow_mut();
        core_row.is_valid = F::ONE;
    }
}

// ========================= ADAPTER ==============================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct DeferralSetupAdapterRecord<F> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub write_aux: [MemoryWriteAuxRecord<F, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(Clone, Copy)]
pub struct DeferralSetupAdapterExecutor;

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
        // Circuit accumulator is stored at the start of the native address space
        debug_assert_eq!(a.as_canonical_u32(), 0);
        debug_assert_eq!(d.as_canonical_u32(), NATIVE_AS);
        for chunk_idx in 0..DIGEST_MEMORY_OPS {
            tracing_write_native(
                memory,
                (chunk_idx * MEMORY_OP_SIZE) as u32,
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
