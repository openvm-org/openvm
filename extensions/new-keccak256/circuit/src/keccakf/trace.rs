
use std::{borrow::BorrowMut, mem::{align_of, size_of}};
use core::convert::TryInto;

use openvm_circuit::{arch::*, system::{memory::online::TracingMemory, poseidon2::trace}};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP};
use openvm_new_keccak256_transpiler::XorinOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, prover::metrics::TraceCells};
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;
use openvm_instructions::riscv::{RV32_REGISTER_AS, RV32_MEMORY_AS};

use crate::{keccakf::KeccakfVmExecutor};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use crate::xorin::XorinVmFiller;
use openvm_circuit::system::memory::MemoryAuxColsFactory;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;

use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix}
};

#[derive(Clone, Copy)]
pub struct KeccakfVmMetadata {
}

impl MultiRowMetadata for KeccakfVmMetadata {
    fn get_num_rows(&self) -> usize {
        1
    }
}

pub(crate) type KeccakfVmRecordLayout = MultiRowLayout<KeccakfVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct KeccakfVmRecordHeader {
    pub pc: u32,
    pub timestamp: u32,
    pub buffer: u32,
    pub preimage_buffer_bytes: [u8; 200],
    pub rd_ptr: u32,
    pub register_aux_cols: [MemoryReadAuxRecord; 1],
    pub buffer_read_aux_cols: [MemoryReadAuxRecord; 200/4],
    pub buffer_write_aux_cols: [MemoryWriteBytesAuxRecord<4>; 200/4],
}

pub struct KeccakfVmRecordMut<'a> {
    pub inner: &'a mut KeccakfVmRecordHeader,
}

impl<'a> CustomBorrow<'a, KeccakfVmRecordMut<'a>, KeccakfVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: KeccakfVmRecordLayout) -> KeccakfVmRecordMut<'a> {
        let (record_buf, _rest) = unsafe {
            self.split_at_mut_unchecked(size_of::<KeccakfVmRecordHeader>())
        };
        KeccakfVmRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> KeccakfVmRecordLayout {
        KeccakfVmRecordLayout {
            metadata: KeccakfVmMetadata {
            },
        }
    }
}

impl<F, RA> PreflightExecutor<F, RA> for KeccakfVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, KeccakfVmRecordLayout, KeccakfVmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", KeccakfOpcode::KECCAKF)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction {a, .. } = instruction;

        let record = state
            .ctx
            .alloc(KeccakfVmRecordLayout::new(KeccakfVmMetadata {  }));

        record.inner.pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.buffer = u32::from_le_bytes(tracing_read(state.memory, RV32_REGISTER_AS, a.as_canonical_u32(), &mut record.inner.register_aux_cols[0].prev_timestamp));

        for idx in 0..(200/4) {
            let read = tracing_read::<4>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * 4) as u32,
                &mut record.inner.buffer_read_aux_cols[idx].prev_timestamp
            );

            record.inner.preimage_buffer_bytes[4*idx..4*(idx+1)].copy_from_slice(&read);
        }

        let preimage_buffer_bytes = record.inner.preimage_buffer_bytes;
        // todo: check if this conversion needs to be constrained
        let mut preimage_buffer_bytes_u64: [u64; 25] = [0; 25];

        // todo: define constants instead of number directly
        for idx in 0..(200/8) {
            preimage_buffer_bytes_u64[idx] = u64::from_le_bytes(
                preimage_buffer_bytes[8 * idx..8 * idx + 8].try_into().unwrap()
            );
        }

        tiny_keccak::keccakf(&mut preimage_buffer_bytes_u64);

        // result is placed in preimage_buffer_bytes_u64
        // convert back to blocks of u8's 
        // todo: think if this conversion needs to be constrained
        let mut result_u8 : [u8; 200] = [0; 200];
        for idx in 0..(200/8) {
            let chunk: [u8; 8] = preimage_buffer_bytes_u64[idx].to_be_bytes();
            result_u8[8 * idx .. 8 * idx + 8].copy_from_slice(&chunk);
        }

        for idx in 0..(200/4) {
            let chunk: [u8; 4] = result_u8[4 * idx .. 4 * idx + 4].try_into().unwrap();
            tracing_write(
                state.memory, 
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * 4) as u32,
                chunk,
                &mut record.inner.buffer_write_aux_cols[idx].prev_timestamp,
                &mut record.inner.buffer_write_aux_cols[idx].prev_data,
            );
        }

        state.memory.timestamp = state.memory.timestamp();
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
