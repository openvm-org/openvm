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
use openvm_instructions::{instruction::Instruction, riscv::RV32_REGISTER_NUM_LIMBS};
use openvm_new_keccak256_transpiler::XorinOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::xorin::{
    columns::XorinVmCols,
    trace::instructions::{
        program::DEFAULT_PC_STEP,
        riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS},
    },
    XorinVmExecutor, XorinVmFiller,
};

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
    pub input_read_aux_cols: [MemoryReadAuxRecord; 34],
    pub buffer_read_aux_cols: [MemoryReadAuxRecord; 34],
    pub buffer_write_aux_cols: [MemoryWriteBytesAuxRecord<4>; 34],
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
        let len = u32::from_le_bytes(unsafe { guest_mem.read::<u8, 4>(1, c.as_canonical_u32()) })
            as usize;
        // Safety: length has to be multiple of 4
        // This is enforced by how the guest program calls the xorin opcode
        // Xorin opcode is only called through the keccak update guest program
        debug_assert!(len.is_multiple_of(4));
        let num_reads = len.div_ceil(4);
        println!("debug preflight num_reads {}", num_reads);

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

        record.inner.buffer = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rd_ptr,
            &mut record.inner.register_aux_cols[0].prev_timestamp,
        ));

        record.inner.input = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs1_ptr,
            &mut record.inner.register_aux_cols[1].prev_timestamp,
        ));

        record.inner.len = u32::from_le_bytes(tracing_read(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.rs2_ptr,
            &mut record.inner.register_aux_cols[2].prev_timestamp,
        ));

        println!("debug record.inner.len {}", record.inner.len);

        debug_assert!(record.inner.buffer as usize + len <= (1 << self.pointer_max_bits));
        debug_assert!(record.inner.input as usize + len < (1 << self.pointer_max_bits));
        debug_assert!(record.inner.len < (1 << self.pointer_max_bits));

        println!("timestamp before buffer read {}", state.memory.timestamp());

        // read buffer
        for idx in 0..num_reads {
            let read = tracing_read::<4>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * 4) as u32,
                &mut record.inner.buffer_read_aux_cols[idx].prev_timestamp,
            );
            record.inner.buffer_limbs[4 * idx..4 * (idx + 1)].copy_from_slice(&read);
        }

        println!("timestamp before input read {}", state.memory.timestamp());

        // read input
        for idx in 0..num_reads {
            let read = tracing_read::<4>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.input + (idx * 4) as u32,
                &mut record.inner.input_read_aux_cols[idx].prev_timestamp,
            );
            record.inner.input_limbs[4 * idx..4 * (idx + 1)].copy_from_slice(&read);
        }

        let mut result = [0u8; 136];

        // execute xorin
        for ((x_xor_y, &x), &y) in result
            .iter_mut()
            .zip(record.inner.buffer_limbs.iter())
            .zip(record.inner.input_limbs.iter())
        {
            *x_xor_y = x ^ y;
        }

        println!("timestamp before input write {}", state.memory.timestamp());

        // write result
        for idx in 0..num_reads {
            let mut word: [u8; 4] = [0u8; 4];
            word.copy_from_slice(&result[4 * idx..4 * (idx + 1)]);
            println!("word = {:?}", word);
            tracing_write(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * 4) as u32,
                word,
                &mut record.inner.buffer_write_aux_cols[idx].prev_timestamp,
                &mut record.inner.buffer_write_aux_cols[idx].prev_data,
            );
        }

        // Todo: use constants instead of number directly
        state.memory.timestamp = state.memory.timestamp();
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        println!("debug record {:?}", record.inner.clone());

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

        trace_row.instruction.pc = F::from_canonical_u32(record.from_pc);
        trace_row.instruction.is_enabled = F::ONE;
        trace_row.instruction.buffer_ptr = F::from_canonical_u32(record.rd_ptr);
        trace_row.instruction.input_ptr = F::from_canonical_u32(record.rs1_ptr);
        trace_row.instruction.len_ptr = F::from_canonical_u32(record.rs2_ptr);
        trace_row.instruction.buffer = F::from_canonical_u32(record.buffer);
        let buffer_u8: [u8; 4] = record.buffer.to_le_bytes();
        let buffer_limbs: [F; 4] = [
            F::from_canonical_u8(buffer_u8[0]),
            F::from_canonical_u8(buffer_u8[1]),
            F::from_canonical_u8(buffer_u8[2]),
            F::from_canonical_u8(buffer_u8[3]),
        ];
        trace_row.instruction.buffer_limbs = buffer_limbs;
        trace_row.instruction.input = F::from_canonical_u32(record.input);
        let input_u8: [u8; 4] = record.input.to_le_bytes();
        let input_limbs: [F; 4] = [
            F::from_canonical_u8(input_u8[0]),
            F::from_canonical_u8(input_u8[1]),
            F::from_canonical_u8(input_u8[2]),
            F::from_canonical_u8(input_u8[3]),
        ];
        trace_row.instruction.input_limbs = input_limbs;
        trace_row.instruction.len = F::from_canonical_u32(record.len);
        let len_u8: [u8; 4] = record.len.to_le_bytes();
        let len_limbs: [F; 4] = [
            F::from_canonical_u8(len_u8[0]),
            F::from_canonical_u8(len_u8[1]),
            F::from_canonical_u8(len_u8[2]),
            F::from_canonical_u8(len_u8[3]),
        ];
        trace_row.instruction.len_limbs = len_limbs;
        trace_row.instruction.start_timestamp = F::from_canonical_u32(record.timestamp);

        for i in 0..(record.len / 4) {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ZERO;
        }
        for i in (record.len / 4)..34 {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ONE;
        }

        let mut timestamp = record.timestamp;
        let record_len: usize = record.len as usize;
        let num_reads: usize = record_len.div_ceil(4);

        // todo: think if the order matters here (maybe due to timestamp things), but this should be
        // fine since it is matched with the one in preflight
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

        // safety note: we leave the upper record_len..134 bytes with zeroes
        // because they are just padding bytes and unused by the chip
        for i in 0..record_len {
            trace_row.sponge.preimage_buffer_bytes[i] =
                F::from_canonical_u8(record.buffer_limbs[i]);
            trace_row.sponge.input_bytes[i] = F::from_canonical_u8(record.input_limbs[i]);
            trace_row.sponge.postimage_buffer_bytes[i] =
                F::from_canonical_u8(record.buffer_limbs[i] ^ record.input_limbs[i]);
            let b_val = record.buffer_limbs[i] as u32;
            let c_val = record.input_limbs[i] as u32;
            self.bitwise_lookup_chip.request_xor(b_val, c_val);
        }

        for t in 0..num_reads {
            mem_helper.fill(
                record.buffer_write_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_write_aux_cols[t].as_mut(),
            );
            trace_row.mem_oc.buffer_bytes_write_aux_cols[t].prev_data = record
                .buffer_write_aux_cols[t]
                .prev_data
                .map(F::from_canonical_u8);
            timestamp += 1;
        }

        let buffer_limbs = record.buffer.to_le_bytes();
        let input_limbs = record.input.to_le_bytes();
        let len_limbs = record.len.to_le_bytes();

        let need_range_check = [
            buffer_limbs.last().unwrap(),
            input_limbs.last().unwrap(),
            len_limbs.last().unwrap(),
            len_limbs.last().unwrap(),
        ];

        let limb_shift = 1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.pointer_max_bits);

        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_chip
                .request_range((pair[0] * limb_shift) as u32, (pair[1] * limb_shift) as u32);
        }
    }
}
