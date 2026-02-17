use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{
        get_record_from_slice, AdapterAirContext, AdapterTraceExecutor, AdapterTraceFiller,
        ExecutionBridge, ExecutionState, ImmInstruction, VmAdapterAir, VmAdapterInterface,
    },
    system::{
        memory::{
            offline_checker::{
                MemoryBridge, MemoryReadAuxCols, MemoryReadAuxRecord, MemoryWriteAuxCols,
                MemoryWriteAuxRecord, MemoryWriteBytesAuxRecord,
            },
            online::TracingMemory,
            MemoryAddress, MemoryAuxColsFactory,
        },
        native_adapter::util::{tracing_read_native, tracing_write_native},
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    NATIVE_AS,
};
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    call::{DeferralCallReads, DeferralCallWrites},
    utils::{bytes_to_f, combine_output, OUTPUT_TOTAL_BYTES},
};

///////////////////////////////////////////////////////////////////////////////////////
/// AIR
///////////////////////////////////////////////////////////////////////////////////////

pub struct DeferralCallAdapterInterface;

impl<T> VmAdapterInterface<T> for DeferralCallAdapterInterface {
    type Reads = DeferralCallReads<T, T>;
    type Writes = DeferralCallWrites<T, T>;
    type ProcessedInstruction = ImmInstruction<T>;
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralCallAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs_ptr: T,

    // Heap pointers and aux columns
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // Read auxiliary columns
    pub input_commit_aux: MemoryReadAuxCols<T>,
    pub old_input_acc_aux: MemoryReadAuxCols<T>,
    pub old_output_acc_aux: MemoryReadAuxCols<T>,

    // Write auxiliary columns
    pub output_commit_and_len_aux: MemoryWriteAuxCols<T, OUTPUT_TOTAL_BYTES>,
    pub new_input_acc_aux: MemoryWriteAuxCols<T, DIGEST_SIZE>,
    pub new_output_acc_aux: MemoryWriteAuxCols<T, DIGEST_SIZE>,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralCallAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    native_start_ptr: u32,
}

impl<F: Field> BaseAir<F> for DeferralCallAdapterAir {
    fn width(&self) -> usize {
        DeferralCallAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for DeferralCallAdapterAir {
    type Interface = DeferralCallAdapterInterface;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &DeferralCallAdapterCols<_> = local.borrow();
        let timestamp = cols.from_state.timestamp;
        let mut timestamp_delta = 0usize;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // Operands a and b are RV32 register pointers. Their values are read first
        // to get heap pointers for output write and input commit read respectively.
        let d = AB::Expr::from_u32(RV32_REGISTER_AS);
        let e = AB::Expr::from_u32(RV32_MEMORY_AS);

        // Heap pointers are first read from their respective registers.
        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), cols.rd_ptr),
                cols.rd_val,
                timestamp_pp(),
                &cols.rd_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), cols.rs_ptr),
                cols.rs_val,
                timestamp_pp(),
                &cols.rs_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // Accumulators are read then updated in the native address space, using
        // deferral_idx (instruction immediate / operand c) to determine the
        // accumulator memory address.
        let input_ptr = bytes_to_f(&cols.rs_val);
        let output_ptr = bytes_to_f(&cols.rd_val);

        let deferral_idx = ctx.instruction.immediate;
        let native_as = AB::Expr::from_u32(NATIVE_AS);

        let digest_size = AB::F::from_usize(DIGEST_SIZE);
        let input_acc_ptr = AB::Expr::from_u32(self.native_start_ptr)
            + (deferral_idx.clone() * AB::Expr::TWO + AB::Expr::ONE) * digest_size;
        let output_acc_ptr = input_acc_ptr.clone() + digest_size;

        self.memory_bridge
            .read(
                MemoryAddress::new(e.clone(), input_ptr),
                ctx.reads.input_commit.clone(),
                timestamp_pp(),
                &cols.input_commit_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), input_acc_ptr.clone()),
                ctx.reads.old_input_acc.clone(),
                timestamp_pp(),
                &cols.old_input_acc_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(native_as.clone(), output_acc_ptr.clone()),
                ctx.reads.old_output_acc.clone(),
                timestamp_pp(),
                &cols.old_output_acc_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        let output_commit_and_len = combine_output(
            ctx.writes.output_commit.clone(),
            ctx.writes.output_len.clone(),
        );
        self.memory_bridge
            .write(
                MemoryAddress::new(e.clone(), output_ptr),
                output_commit_and_len,
                timestamp_pp(),
                &cols.output_commit_and_len_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), input_acc_ptr),
                ctx.writes.new_input_acc.clone(),
                timestamp_pp(),
                &cols.new_input_acc_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(native_as.clone(), output_acc_ptr),
                ctx.writes.new_output_acc.clone(),
                timestamp_pp(),
                &cols.new_output_acc_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    cols.rd_ptr.into(),
                    cols.rs_ptr.into(),
                    deferral_idx,
                    d.clone(),
                    e.clone(),
                ],
                cols.from_state,
                AB::Expr::from_usize(timestamp_delta),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &DeferralCallAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

///////////////////////////////////////////////////////////////////////////////////////
/// EXECUTION + TRACEGEN
///////////////////////////////////////////////////////////////////////////////////////

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
    pub input_commit_aux: MemoryReadAuxRecord,
    pub old_input_acc_aux: MemoryReadAuxRecord,
    pub old_output_acc_aux: MemoryReadAuxRecord,

    // Write auxiliary records
    pub output_commit_and_len_aux: MemoryWriteBytesAuxRecord<OUTPUT_TOTAL_BYTES>,
    pub new_input_acc_aux: MemoryWriteAuxRecord<F, DIGEST_SIZE>,
    pub new_output_acc_aux: MemoryWriteAuxRecord<F, DIGEST_SIZE>,
}

#[derive(derive_new::new, Clone, Copy)]
pub struct DeferralCallAdapterExecutor {
    pub(crate) native_start_ptr: u32,
}

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

        let input_commit = tracing_read(
            memory,
            e.as_canonical_u32(),
            u32::from_le_bytes(record.rs_val),
            &mut record.input_commit_aux.prev_timestamp,
        );

        let deferral_idx = c.as_canonical_u32();
        let input_acc_ptr = self.native_start_ptr + (2 * deferral_idx + 1) * (DIGEST_SIZE as u32);
        let output_acc_ptr = input_acc_ptr + (DIGEST_SIZE as u32);

        let old_input_acc = tracing_read_native(
            memory,
            input_acc_ptr,
            &mut record.old_input_acc_aux.prev_timestamp,
        );
        let old_output_acc = tracing_read_native(
            memory,
            output_acc_ptr,
            &mut record.old_output_acc_aux.prev_timestamp,
        );

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

        let output_commit_and_len = combine_output(data.output_commit, data.output_len);
        tracing_write(
            memory,
            e.as_canonical_u32(),
            u32::from_le_bytes(record.rd_val),
            output_commit_and_len,
            &mut record.output_commit_and_len_aux.prev_timestamp,
            &mut record.output_commit_and_len_aux.prev_data,
        );

        let deferral_idx = c.as_canonical_u32();
        let input_acc_ptr = self.native_start_ptr + (2 * deferral_idx + 1) * (DIGEST_SIZE as u32);
        let output_acc_ptr = input_acc_ptr + (DIGEST_SIZE as u32);

        tracing_write_native(
            memory,
            input_acc_ptr,
            data.new_input_acc,
            &mut record.new_input_acc_aux.prev_timestamp,
            &mut record.new_input_acc_aux.prev_data,
        );

        tracing_write_native(
            memory,
            output_acc_ptr,
            data.new_output_acc,
            &mut record.new_output_acc_aux.prev_timestamp,
            &mut record.new_output_acc_aux.prev_data,
        );
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

        // Writing in reverse order to avoid overwriting the record
        mem_helper.fill(
            record.new_output_acc_aux.prev_timestamp,
            record.new_output_acc_aux.prev_timestamp + 1,
            adapter_row.new_output_acc_aux.as_mut(),
        );
        mem_helper.fill(
            record.new_input_acc_aux.prev_timestamp,
            record.new_input_acc_aux.prev_timestamp + 1,
            adapter_row.new_input_acc_aux.as_mut(),
        );
        mem_helper.fill(
            record.output_commit_and_len_aux.prev_timestamp,
            record.output_commit_and_len_aux.prev_timestamp + 1,
            adapter_row.output_commit_and_len_aux.as_mut(),
        );

        mem_helper.fill(
            record.old_output_acc_aux.prev_timestamp,
            record.old_output_acc_aux.prev_timestamp + 1,
            adapter_row.old_output_acc_aux.as_mut(),
        );
        mem_helper.fill(
            record.old_input_acc_aux.prev_timestamp,
            record.old_input_acc_aux.prev_timestamp + 1,
            adapter_row.old_input_acc_aux.as_mut(),
        );
        mem_helper.fill(
            record.input_commit_aux.prev_timestamp,
            record.input_commit_aux.prev_timestamp + 1,
            adapter_row.input_commit_aux.as_mut(),
        );

        mem_helper.fill(
            record.rs_aux.prev_timestamp,
            record.rs_aux.prev_timestamp + 1,
            adapter_row.rs_aux.as_mut(),
        );
        mem_helper.fill(
            record.rd_aux.prev_timestamp,
            record.rd_aux.prev_timestamp + 1,
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
