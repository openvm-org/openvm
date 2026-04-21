use std::{
    borrow::BorrowMut,
    mem::{align_of, size_of},
};

use openvm_circuit::{
    arch::*,
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
        MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_WORD_NUM_LIMBS},
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_riscv_circuit::adapters::{read_rv64_register, tracing_read, tracing_write};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::xorin::{columns::XorinVmCols, XorinVmExecutor, XorinVmFiller};

#[derive(Clone, Copy)]
pub struct XorinVmMetadata {}

impl MultiRowMetadata for XorinVmMetadata {
    fn get_num_rows(&self) -> usize {
        1
    }
}

pub(crate) type XorinVmRecordLayout = MultiRowLayout<XorinVmMetadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct XorinVmRecordHeader {
    pub from_pc: u32,
    pub timestamp: u32,
    pub rd_ptr: u32,
    pub rs1_ptr: u32,
    pub rs2_ptr: u32,
    pub buffer: u32,
    pub input: u32,
    pub len: u32,
    pub buffer_limbs: [u8; 136],
    pub input_limbs: [u8; 136],
    pub register_aux_cols: [MemoryReadAuxRecord; 3],
    pub input_read_aux_cols: [MemoryReadAuxRecord; 17],
    pub buffer_read_aux_cols: [MemoryReadAuxRecord; 17],
    pub buffer_write_aux_cols: [MemoryWriteBytesAuxRecord<8>; 17],
}

pub struct XorinVmRecordMut<'a> {
    pub inner: &'a mut XorinVmRecordHeader,
}

// Custom borrowing to split the buffer into a fixed `XorinVmRecord` header
impl<'a> CustomBorrow<'a, XorinVmRecordMut<'a>, XorinVmRecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, _layout: XorinVmRecordLayout) -> XorinVmRecordMut<'a> {
        let (record_buf, _rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<XorinVmRecordHeader>()) };
        XorinVmRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> XorinVmRecordLayout {
        XorinVmRecordLayout {
            metadata: XorinVmMetadata {},
        }
    }
}

impl SizedRecord<XorinVmRecordLayout> for XorinVmRecordMut<'_> {
    fn size(_layout: &XorinVmRecordLayout) -> usize {
        size_of::<XorinVmRecordHeader>()
    }

    fn alignment(_layout: &XorinVmRecordLayout) -> usize {
        align_of::<XorinVmRecordHeader>()
    }
}

impl<F, RA> PreflightExecutor<F, RA> for XorinVmExecutor
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, XorinVmRecordLayout, XorinVmRecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", XorinOpcode::XORIN)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, b, c, .. } = instruction;

        // Reading the length first without tracing to allocate a record of correct size
        let guest_mem = state.memory.data();
        let len = read_rv64_register(guest_mem, c.as_canonical_u32()) as u32 as usize;
        // Safety: length has to be multiple of 4
        // This is enforced by how the guest program calls the xorin opcode
        // Xorin opcode is only called through the keccak update guest program
        debug_assert!(len.is_multiple_of(4));
        let num_reads = len.div_ceil(8);

        // safety: the below alloc uses MultiRowLayout alloc implementation because
        // XorinVmRecordLayout is a MultiRowLayout since get_num_rows() = 1, this will
        // alloc_buffer of size width where width is the width of the trace matrix
        // then it takes a prefix of this allocated buffer through custom borrow
        // of length XorinVmRecordLayout size and return it as the below `record`
        let record = state
            .ctx
            .alloc(XorinVmRecordLayout::new(XorinVmMetadata {}));

        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

        let buffer_val: [u8; 8] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            record.inner.rd_ptr,
            &mut record.inner.register_aux_cols[0].prev_timestamp,
        );
        record.inner.buffer = u32::from_le_bytes(buffer_val[..4].try_into().unwrap());

        let input_val: [u8; 8] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            record.inner.rs1_ptr,
            &mut record.inner.register_aux_cols[1].prev_timestamp,
        );
        record.inner.input = u32::from_le_bytes(input_val[..4].try_into().unwrap());

        let len_val: [u8; 8] = tracing_read(
            state.memory,
            RV64_REGISTER_AS,
            record.inner.rs2_ptr,
            &mut record.inner.register_aux_cols[2].prev_timestamp,
        );
        record.inner.len = u32::from_le_bytes(len_val[..4].try_into().unwrap());

        debug_assert!(record.inner.buffer as usize + len <= (1 << self.pointer_max_bits));
        debug_assert!(record.inner.input as usize + len < (1 << self.pointer_max_bits));
        debug_assert!(record.inner.len < (1 << self.pointer_max_bits));

        // read buffer in 8-byte blocks
        for idx in 0..num_reads {
            let read = tracing_read::<8>(
                state.memory,
                RV64_MEMORY_AS,
                record.inner.buffer + (idx * 8) as u32,
                &mut record.inner.buffer_read_aux_cols[idx].prev_timestamp,
            );
            record.inner.buffer_limbs[8 * idx..8 * (idx + 1)].copy_from_slice(&read);
        }

        // read input in 8-byte blocks
        for idx in 0..num_reads {
            let read = tracing_read::<8>(
                state.memory,
                RV64_MEMORY_AS,
                record.inner.input + (idx * 8) as u32,
                &mut record.inner.input_read_aux_cols[idx].prev_timestamp,
            );
            record.inner.input_limbs[8 * idx..8 * (idx + 1)].copy_from_slice(&read);
        }

        let mut result = [0u8; 136];

        // execute xorin — only XOR the first `len` active bytes
        // Padding bytes in boundary 8-byte blocks keep the original buffer value
        result[..len].copy_from_slice(&record.inner.buffer_limbs[..len]);
        for i in 0..len {
            result[i] ^= record.inner.input_limbs[i];
        }
        // For the boundary block, fill remaining bytes with original buffer data
        let bytes_covered = num_reads * 8;
        result[len..bytes_covered]
            .copy_from_slice(&record.inner.buffer_limbs[len..bytes_covered]);

        // write result in 8-byte blocks
        for idx in 0..num_reads {
            let mut word: [u8; 8] = [0u8; 8];
            word.copy_from_slice(&result[8 * idx..8 * (idx + 1)]);
            tracing_write(
                state.memory,
                RV64_MEMORY_AS,
                record.inner.buffer + (idx * 8) as u32,
                word,
                &mut record.inner.buffer_write_aux_cols[idx].prev_timestamp,
                &mut record.inner.buffer_write_aux_cols[idx].prev_data,
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for XorinVmFiller {
    fn fill_trace_row(&self, mem_helper: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        let record: XorinVmRecordMut = unsafe {
            get_record_from_slice(
                &mut row_slice,
                XorinVmRecordLayout {
                    metadata: XorinVmMetadata {},
                },
            )
        };

        // Safety: the clone here is necessary because the XorinVmCols uses the same buffer
        let record = record.inner.clone();
        row_slice.fill(F::ZERO);
        let trace_row: &mut XorinVmCols<F> = row_slice.borrow_mut();

        trace_row.instruction.pc = F::from_u32(record.from_pc);
        trace_row.instruction.is_enabled = F::ONE;
        trace_row.instruction.buffer_reg_ptr = F::from_u32(record.rd_ptr);
        trace_row.instruction.input_reg_ptr = F::from_u32(record.rs1_ptr);
        trace_row.instruction.len_reg_ptr = F::from_u32(record.rs2_ptr);
        trace_row.instruction.buffer_ptr = F::from_u32(record.buffer);
        trace_row.instruction.buffer_ptr_limbs = record.buffer.to_le_bytes().map(F::from_u8);
        trace_row.instruction.input_ptr = F::from_u32(record.input);
        trace_row.instruction.input_ptr_limbs = record.input.to_le_bytes().map(F::from_u8);
        trace_row.instruction.len = F::from_u32(record.len);
        trace_row.instruction.len_limbs = record.len.to_le_bytes().map(F::from_u8);
        trace_row.instruction.start_timestamp = F::from_u32(record.timestamp);

        for i in 0..(record.len / 4) {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ZERO;
        }
        for i in (record.len / 4)..34 {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ONE;
        }

        let mut timestamp = record.timestamp;
        let record_len: usize = record.len as usize;
        let num_reads: usize = record_len.div_ceil(8);

        for t in 0..3 {
            mem_helper.fill(
                record.register_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.register_aux_cols[t].as_mut(),
            );

            timestamp += 1;
        }

        for t in 0..num_reads {
            mem_helper.fill(
                record.buffer_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_read_aux_cols[t].as_mut(),
            );
            timestamp += 1;
        }

        for t in 0..num_reads {
            mem_helper.fill(
                record.input_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.input_bytes_read_aux_cols[t].as_mut(),
            );
            timestamp += 1;
        }

        // Fill all bytes that are covered by active 8-byte memory blocks.
        // For non-padding bytes, postimage = preimage XOR input.
        // For padding bytes within an active block, postimage = preimage (identity).
        let bytes_covered = num_reads * 8;
        for i in 0..record_len {
            trace_row.sponge.preimage_buffer_bytes[i] = F::from_u8(record.buffer_limbs[i]);
            trace_row.sponge.input_bytes[i] = F::from_u8(record.input_limbs[i]);
            trace_row.sponge.postimage_buffer_bytes[i] =
                F::from_u8(record.buffer_limbs[i] ^ record.input_limbs[i]);
            let b_val = record.buffer_limbs[i] as u32;
            let c_val = record.input_limbs[i] as u32;
            self.bitwise_lookup_chip.request_xor(b_val, c_val);
        }
        // Padding bytes within active blocks: postimage = preimage
        for i in record_len..bytes_covered {
            trace_row.sponge.preimage_buffer_bytes[i] = F::from_u8(record.buffer_limbs[i]);
            trace_row.sponge.input_bytes[i] = F::from_u8(record.input_limbs[i]);
            trace_row.sponge.postimage_buffer_bytes[i] = F::from_u8(record.buffer_limbs[i]);
        }

        for t in 0..num_reads {
            mem_helper.fill(
                record.buffer_write_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_write_aux_cols[t].as_mut(),
            );
            trace_row.mem_oc.buffer_bytes_write_aux_cols[t].prev_data =
                record.buffer_write_aux_cols[t].prev_data.map(F::from_u8);
            timestamp += 1;
        }

        let msb_byte = |val: u32| -> u32 { (val >> (RV64_CELL_BITS * (RV64_WORD_NUM_LIMBS - 1))) & 0xFF };
        let need_range_check = [
            msb_byte(record.buffer),
            msb_byte(record.input),
            msb_byte(record.len),
            msb_byte(record.len),
        ];

        let limb_shift = 1u32 << (RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.pointer_max_bits);

        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range(pair[0] * limb_shift, pair[1] * limb_shift);
        }
    }
}
