use std::{array::from_fn, borrow::BorrowMut, sync::Arc};

use itertools::Itertools;
use openvm_circuit::{
    arch::{
        get_record_from_slice,
        hasher::{Hasher, HasherChip},
        AdapterTraceExecutor, AdapterTraceFiller, EmptyAdapterCoreLayout, ExecutionError,
        PreflightExecutor, RecordArena, TraceFiller, VmField, VmStateMut,
    },
    system::{
        memory::{
            offline_checker::{
                MemoryReadAuxRecord, MemoryWriteAuxRecord, MemoryWriteBytesAuxRecord,
            },
            online::TracingMemory,
            MemoryAuxColsFactory,
        },
        native_adapter::util::{tracing_read_native, tracing_write_native},
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    call::{DeferralCallAdapterCols, DeferralCallCoreCols, DeferralCallReads, DeferralCallWrites},
    count::DeferralCircuitCountChip,
    poseidon2::{deferral_poseidon2_chip, DeferralPoseidon2Chip},
    utils::{
        byte_commit_to_f, combine_output, join_memory_ops, memory_op_chunk, COMMIT_MEMORY_OPS,
        DIGEST_MEMORY_OPS, F_NUM_BYTES, MEMORY_OP_SIZE, OUTPUT_TOTAL_MEMORY_OPS,
    },
    DeferralFn,
};

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct DeferralCallCoreRecord<F> {
    pub deferral_idx: F,
    pub read_data: DeferralCallReads<u8, F>,
    pub write_data: DeferralCallWrites<u8, F>,
}

#[derive(Clone, derive_new::new)]
pub struct DeferralCallCoreExecutor<A> {
    pub(in crate::call) adapter: A,
    pub(in crate::call) deferral_fns: Vec<Arc<DeferralFn>>,
}

#[derive(Clone, Debug, derive_new::new)]
pub struct DeferralCallCoreFiller<A, F: VmField> {
    adapter: A,
    count_chip: Arc<DeferralCircuitCountChip>,
    poseidon2_chip: Arc<DeferralPoseidon2Chip<F>>,
}

impl<F, A, RA> PreflightExecutor<F, RA> for DeferralCallCoreExecutor<A>
where
    F: VmField,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = DeferralCallReads<u8, F>,
            WriteData = DeferralCallWrites<u8, F>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut DeferralCallCoreRecord<F>),
    >,
{
    fn get_opcode_name(&self, _opcode: usize) -> String {
        format!("{:?}", DeferralOpcode::CALL)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());
        A::start(*state.pc, state.memory, &mut adapter_record);
        core_record.deferral_idx = instruction.c;

        let read_data = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);
        core_record.read_data = read_data;

        let input_commit = byte_commit_to_f(&read_data.input_commit.map(F::from_u8));
        let def_idx = instruction.c.as_canonical_u32();
        let poseidon2_chip = deferral_poseidon2_chip();

        let (output_commit, output_len) = self.deferral_fns[def_idx as usize].execute(
            &read_data.input_commit.to_vec(),
            &mut state.streams.deferrals[def_idx as usize],
            def_idx,
            &poseidon2_chip,
        );

        let output_f_commit =
            byte_commit_to_f(&output_commit.iter().map(|v| F::from_u8(*v)).collect_vec());
        let new_input_acc = poseidon2_chip.compress(&read_data.old_input_acc, &input_commit);
        let new_output_acc = poseidon2_chip.compress(&read_data.old_output_acc, &output_f_commit);

        let output_len_u32 =
            u32::try_from(output_len).expect("deferral output length should fit in a u32");
        let write_data = DeferralCallWrites {
            output_commit: output_commit.try_into().unwrap(),
            output_len: output_len_u32.to_le_bytes(),
            new_input_acc,
            new_output_acc,
        };
        core_record.write_data = write_data;
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A> TraceFiller<F> for DeferralCallCoreFiller<A, F>
where
    F: VmField,
    A: 'static + AdapterTraceFiller<F>,
{
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, row_slice: &mut [F]) {
        // SAFETY: row_slice is guaranteed by the caller to have at least A::WIDTH +
        // DeferralCallCoreCols::width() elements
        let (adapter_row, mut core_row) = unsafe { row_slice.split_at_mut_unchecked(A::WIDTH) };
        self.adapter.fill_trace_row(mem_helper, adapter_row);

        // SAFETY: core_row contains a valid DeferralCallCoreRecord written by the executor
        // during trace generation
        let record: &DeferralCallCoreRecord<F> =
            unsafe { get_record_from_slice(&mut core_row, ()) };
        let cols: &mut DeferralCallCoreCols<F> = core_row.borrow_mut();

        self.count_chip
            .add_count(record.deferral_idx.as_canonical_u32());

        let input_f_commit: [F; _] =
            byte_commit_to_f(&record.read_data.input_commit.map(F::from_u8));
        let output_f_commit: [F; _] =
            byte_commit_to_f(&record.write_data.output_commit.map(F::from_u8));
        self.poseidon2_chip
            .compress_and_record(&record.read_data.old_input_acc, &input_f_commit);
        self.poseidon2_chip
            .compress_and_record(&record.read_data.old_output_acc, &output_f_commit);

        // Write columns in reverse order to avoid clobbering the record.
        cols.writes.new_output_acc = record.write_data.new_output_acc;
        cols.writes.new_input_acc = record.write_data.new_input_acc;
        cols.writes.output_len = record.write_data.output_len.map(F::from_u8);
        cols.writes.output_commit = record.write_data.output_commit.map(F::from_u8);
        cols.reads.old_output_acc = record.read_data.old_output_acc;
        cols.reads.old_input_acc = record.read_data.old_input_acc;
        cols.reads.input_commit = record.read_data.input_commit.map(F::from_u8);
        cols.deferral_idx = record.deferral_idx;
        cols.is_valid = F::ONE;
    }
}

// ========================= ADAPTER ==============================

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct DeferralCallAdapterRecord<F> {
    pub from_pc: u32,
    pub from_timestamp: u32,
    pub rd_ptr: F,
    pub rs_ptr: F,

    // Heap pointers and auxiliary records
    pub rd_val: [u8; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [u8; RV32_REGISTER_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxRecord,
    pub rs_aux: MemoryReadAuxRecord,

    // Read auxiliary records
    pub input_commit_aux: [MemoryReadAuxRecord; COMMIT_MEMORY_OPS],
    pub old_input_acc_aux: [MemoryReadAuxRecord; DIGEST_MEMORY_OPS],
    pub old_output_acc_aux: [MemoryReadAuxRecord; DIGEST_MEMORY_OPS],

    // Write auxiliary records
    pub output_commit_and_len_aux:
        [MemoryWriteBytesAuxRecord<MEMORY_OP_SIZE>; OUTPUT_TOTAL_MEMORY_OPS],
    pub new_input_acc_aux: [MemoryWriteAuxRecord<F, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
    pub new_output_acc_aux: [MemoryWriteAuxRecord<F, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(Clone, Copy)]
pub struct DeferralCallAdapterExecutor;

#[derive(derive_new::new)]
pub struct DeferralCallAdapterFiller;

impl<F: PrimeField32> AdapterTraceExecutor<F> for DeferralCallAdapterExecutor {
    const WIDTH: usize = DeferralCallAdapterCols::<u8>::width();
    type ReadData = DeferralCallReads<u8, F>;
    type WriteData = DeferralCallWrites<u8, F>;
    type RecordMut<'a> = &'a mut DeferralCallAdapterRecord<F>;

    fn start(pc: u32, memory: &TracingMemory, record: &mut Self::RecordMut<'_>) {
        record.from_pc = pc;
        record.from_timestamp = memory.timestamp;
    }

    fn read(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        record: &mut Self::RecordMut<'_>,
    ) -> Self::ReadData {
        let &Instruction { a, b, c, d, e, .. } = instruction;
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);
        record.rd_ptr = a;
        record.rs_ptr = b;

        record.rd_val = tracing_read(
            memory,
            d.as_canonical_u32(),
            a.as_canonical_u32(),
            &mut record.rd_aux.prev_timestamp,
        );
        record.rs_val = tracing_read(
            memory,
            d.as_canonical_u32(),
            b.as_canonical_u32(),
            &mut record.rs_aux.prev_timestamp,
        );

        let input_commit_chunks: [[u8; MEMORY_OP_SIZE]; COMMIT_MEMORY_OPS] = from_fn(|i| {
            tracing_read(
                memory,
                e.as_canonical_u32(),
                u32::from_le_bytes(record.rs_val) + (i * MEMORY_OP_SIZE) as u32,
                &mut record.input_commit_aux[i].prev_timestamp,
            )
        });
        let input_commit = join_memory_ops(input_commit_chunks);

        let deferral_idx = c.as_canonical_u32();
        let input_acc_ptr = (2 * deferral_idx + 1) * (DIGEST_SIZE as u32);
        let output_acc_ptr = input_acc_ptr + (DIGEST_SIZE as u32);

        let old_input_acc_chunks: [[F; MEMORY_OP_SIZE]; DIGEST_MEMORY_OPS] = from_fn(|i| {
            tracing_read_native(
                memory,
                input_acc_ptr + (i * MEMORY_OP_SIZE) as u32,
                &mut record.old_input_acc_aux[i].prev_timestamp,
            )
        });
        let old_output_acc_chunks: [[F; MEMORY_OP_SIZE]; DIGEST_MEMORY_OPS] = from_fn(|i| {
            tracing_read_native(
                memory,
                output_acc_ptr + (i * MEMORY_OP_SIZE) as u32,
                &mut record.old_output_acc_aux[i].prev_timestamp,
            )
        });
        let old_input_acc = join_memory_ops(old_input_acc_chunks);
        let old_output_acc = join_memory_ops(old_output_acc_chunks);

        DeferralCallReads {
            input_commit,
            old_input_acc,
            old_output_acc,
        }
    }

    fn write(
        &self,
        memory: &mut TracingMemory,
        instruction: &Instruction<F>,
        data: Self::WriteData,
        record: &mut Self::RecordMut<'_>,
    ) {
        let &Instruction { c, e, .. } = instruction;
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let output_len_full = from_fn(|i| {
            if i < F_NUM_BYTES {
                data.output_len[i]
            } else {
                0u8
            }
        });

        let output_commit_and_len = combine_output(data.output_commit, output_len_full);
        for chunk_idx in 0..OUTPUT_TOTAL_MEMORY_OPS {
            tracing_write(
                memory,
                e.as_canonical_u32(),
                u32::from_le_bytes(record.rd_val) + (chunk_idx * MEMORY_OP_SIZE) as u32,
                memory_op_chunk(&output_commit_and_len, chunk_idx),
                &mut record.output_commit_and_len_aux[chunk_idx].prev_timestamp,
                &mut record.output_commit_and_len_aux[chunk_idx].prev_data,
            );
        }

        let deferral_idx = c.as_canonical_u32();
        let input_acc_ptr = (2 * deferral_idx + 1) * (DIGEST_SIZE as u32);
        let output_acc_ptr = input_acc_ptr + (DIGEST_SIZE as u32);

        for chunk_idx in 0..DIGEST_MEMORY_OPS {
            tracing_write_native(
                memory,
                input_acc_ptr + (chunk_idx * MEMORY_OP_SIZE) as u32,
                memory_op_chunk(&data.new_input_acc, chunk_idx),
                &mut record.new_input_acc_aux[chunk_idx].prev_timestamp,
                &mut record.new_input_acc_aux[chunk_idx].prev_data,
            );
        }

        for chunk_idx in 0..DIGEST_MEMORY_OPS {
            tracing_write_native(
                memory,
                output_acc_ptr + (chunk_idx * MEMORY_OP_SIZE) as u32,
                memory_op_chunk(&data.new_output_acc, chunk_idx),
                &mut record.new_output_acc_aux[chunk_idx].prev_timestamp,
                &mut record.new_output_acc_aux[chunk_idx].prev_data,
            );
        }
    }
}

impl<F: PrimeField32> AdapterTraceFiller<F> for DeferralCallAdapterFiller {
    const WIDTH: usize = DeferralCallAdapterCols::<u8>::width();

    #[inline(always)]
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut adapter_row: &mut [F]) {
        // SAFETY: caller ensures `adapter_row` contains a valid record representation
        // that was previously written by the executor
        let record: &DeferralCallAdapterRecord<F> =
            unsafe { get_record_from_slice(&mut adapter_row, ()) };
        let adapter_row: &mut DeferralCallAdapterCols<F> = adapter_row.borrow_mut();

        // Timestamps in AIR are assigned in strict sequence starting from
        // `from_state.timestamp`; mirror that exact sequence in reverse here.
        let timestamp_delta =
            2 + COMMIT_MEMORY_OPS + OUTPUT_TOTAL_MEMORY_OPS + 4 * DIGEST_MEMORY_OPS;
        let mut timestamp = record.from_timestamp + timestamp_delta as u32;
        let mut timestamp_mm = || {
            timestamp -= 1;
            timestamp
        };

        // Writing in reverse order to avoid overwriting the record
        for chunk_idx in (0..DIGEST_MEMORY_OPS).rev() {
            adapter_row.new_output_acc_aux[chunk_idx]
                .set_prev_data(record.new_output_acc_aux[chunk_idx].prev_data);
            mem_helper.fill(
                record.new_output_acc_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.new_output_acc_aux[chunk_idx].as_mut(),
            );
        }
        for chunk_idx in (0..DIGEST_MEMORY_OPS).rev() {
            adapter_row.new_input_acc_aux[chunk_idx]
                .set_prev_data(record.new_input_acc_aux[chunk_idx].prev_data);
            mem_helper.fill(
                record.new_input_acc_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.new_input_acc_aux[chunk_idx].as_mut(),
            );
        }
        for chunk_idx in (0..OUTPUT_TOTAL_MEMORY_OPS).rev() {
            adapter_row.output_commit_and_len_aux[chunk_idx].set_prev_data(
                record.output_commit_and_len_aux[chunk_idx]
                    .prev_data
                    .map(F::from_u8),
            );
            mem_helper.fill(
                record.output_commit_and_len_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.output_commit_and_len_aux[chunk_idx].as_mut(),
            );
        }

        for chunk_idx in (0..DIGEST_MEMORY_OPS).rev() {
            mem_helper.fill(
                record.old_output_acc_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.old_output_acc_aux[chunk_idx].as_mut(),
            );
        }
        for chunk_idx in (0..DIGEST_MEMORY_OPS).rev() {
            mem_helper.fill(
                record.old_input_acc_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.old_input_acc_aux[chunk_idx].as_mut(),
            );
        }
        for chunk_idx in (0..COMMIT_MEMORY_OPS).rev() {
            mem_helper.fill(
                record.input_commit_aux[chunk_idx].prev_timestamp,
                timestamp_mm(),
                adapter_row.input_commit_aux[chunk_idx].as_mut(),
            );
        }

        mem_helper.fill(
            record.rs_aux.prev_timestamp,
            timestamp_mm(),
            adapter_row.rs_aux.as_mut(),
        );
        mem_helper.fill(
            record.rd_aux.prev_timestamp,
            timestamp_mm(),
            adapter_row.rd_aux.as_mut(),
        );
        adapter_row.rs_val = record.rs_val.map(F::from_u8);
        adapter_row.rd_val = record.rd_val.map(F::from_u8);

        adapter_row.rs_ptr = record.rs_ptr;
        adapter_row.rd_ptr = record.rd_ptr;
        adapter_row.from_state.timestamp = F::from_u32(record.from_timestamp);
        adapter_row.from_state.pc = F::from_u32(record.from_pc);
    }
}
