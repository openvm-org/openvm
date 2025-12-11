use std::{borrow::BorrowMut, mem::{align_of, size_of}};

use openvm_circuit::{arch::*, system::{memory::online::TracingMemory, poseidon2::trace}};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_new_keccak256_transpiler::Rv32NewKeccakOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::{p3_field::PrimeField32, prover::metrics::TraceCells};
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;
use crate::xorin::{columns::XorinVmCols, trace::instructions::riscv::RV32_REGISTER_AS};
use crate::xorin::trace::instructions::riscv::RV32_MEMORY_AS;
use crate::xorin::trace::instructions::program::DEFAULT_PC_STEP;

use crate::xorin::XorinVmExecutor;
use crate::xorin::XorinVmFiller;
use openvm_circuit::system::memory::MemoryAuxColsFactory;

use openvm_stark_backend::{
    p3_matrix::{dense::RowMajorMatrix}
};

#[derive(Clone, Copy)]
pub struct XorinVmMetadata {
}

impl MultiRowMetadata for XorinVmMetadata {
    // todo: confirm that this is the number of rows in one opcode execute
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
        let (record_buf, _rest) = unsafe {
            self.split_at_mut_unchecked(size_of::<XorinVmRecordHeader>())
        };
        XorinVmRecordMut {
            inner: record_buf.borrow_mut(),
        }
    }

    unsafe fn extract_layout(&self) -> XorinVmRecordLayout {
        XorinVmRecordLayout {
            metadata: XorinVmMetadata {
            },
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
        format!("{:?}", Rv32NewKeccakOpcode::XORIN)
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let &Instruction { a, b, c, .. } = instruction;

        // Reading the length first without tracing to allocate a record of correct size
        let guest_mem = state.memory.data();
        let len = u32::from_le_bytes(unsafe {
            guest_mem.read::<u8, 4>(1, c.as_canonical_u32()) 
        }) as usize;

        println!("len = {}", len);
        // Safety: length has to be multiple of 4
        // This is enforced by how the guest program calls the xorin opcode
        // Xorin opcode is only called through the keccak update guest program
        debug_assert!(len.is_multiple_of(4));
        let num_reads = len.div_ceil(4);

        println!("instruction which called xorin {:?}", instruction);

        // safety: the below alloc uses MultiRowLayout alloc implementation because XorinVmRecordLayout is a MultiRowLayout
        // since get_num_rows() = 1, this will alloc_buffer of size width 
        // where width is the width of the trace matrix 
        // then it takes a prefix of this allocated buffer through custom borrow
        // of length XorinVmRecordLayout size and return it as the below `record`
        let record = state
            .ctx 
            .alloc(XorinVmRecordLayout::new(XorinVmMetadata { }));
        
        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.rd_ptr = a.as_canonical_u32();
        record.inner.rs1_ptr = b.as_canonical_u32();
        record.inner.rs2_ptr = c.as_canonical_u32();

        record.inner.buffer = u32::from_le_bytes(tracing_read(
            state.memory, 
            RV32_REGISTER_AS, 
            record.inner.rd_ptr, 
            &mut record.inner.register_aux_cols[0].prev_timestamp
        ));

        record.inner.input = u32::from_le_bytes(tracing_read(
            state.memory, 
            RV32_REGISTER_AS, 
            record.inner.rs1_ptr, 
            &mut record.inner.register_aux_cols[1].prev_timestamp
        ));

        record.inner.len = u32::from_le_bytes(tracing_read(
            state.memory, 
            RV32_REGISTER_AS, 
            record.inner.rs2_ptr, 
            &mut record.inner.register_aux_cols[2].prev_timestamp
        ));

        println!("record.inner.len = {}", record.inner.len);

        debug_assert!(record.inner.buffer as usize + len <= (1 << self.pointer_max_bits));
        debug_assert!(record.inner.input as usize + len < (1 << self.pointer_max_bits));
        debug_assert!(record.inner.len < (1 << self.pointer_max_bits));

        // read buffer
        for idx in 0..num_reads {
            let read = tracing_read::<4>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.buffer + (idx * 4) as u32, 
                &mut record.inner.buffer_read_aux_cols[idx].prev_timestamp
            );
            record.inner.buffer_limbs[4*idx..4*(idx+1)].copy_from_slice(&read);
        }        

        // read input 
        for idx in 0..num_reads {
            let read = tracing_read::<4>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.input + (idx * 4) as u32, 
                &mut record.inner.input_read_aux_cols[idx].prev_timestamp
            );
            record.inner.input_limbs[4*idx..4*(idx+1)].copy_from_slice(&read);
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

        for (i, word) in result.chunks_exact(4).enumerate() {
            tracing_write(
                state.memory, 
                RV32_MEMORY_AS, 
                record.inner.buffer + (i * 4) as u32, 
                word.try_into().unwrap(), 
                &mut record.inner.buffer_write_aux_cols[i].prev_timestamp, 
                &mut record.inner.buffer_write_aux_cols[i].prev_data 
            );
        }

        // Due to the AIR constraints, the final memory timestamp should be the following 
        // Todo: use constants instead of number directly
        state.memory.timestamp = record.inner.timestamp + 105;
        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}

impl<F: PrimeField32> TraceFiller<F> for XorinVmFiller {
    fn fill_trace_row(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        mut row_slice: &mut [F],
    ) {
        let record: XorinVmRecordMut = unsafe {
            get_record_from_slice(&mut row_slice, XorinVmRecordLayout {
                metadata: XorinVmMetadata {
                }
            })
        };

        let trace_row: &mut XorinVmCols<F> = row_slice.borrow_mut();
        let record = record.inner.borrow_mut();
        trace_row.instruction.pc = F::from_canonical_u32(record.from_pc);
        trace_row.instruction.start_timestamp = F::from_canonical_u32(record.timestamp);
        trace_row.instruction.buffer_ptr = F::from_canonical_u32(record.rd_ptr);
        trace_row.instruction.input_ptr = F::from_canonical_u32(record.rs1_ptr);
        trace_row.instruction.len_ptr = F::from_canonical_u32(record.rs2_ptr);
        let buffer_u8: [u8; 4] = record.buffer.to_le_bytes();
        let buffer_limbs: [F; 4] = [
            F::from_canonical_u8(buffer_u8[0]), 
            F::from_canonical_u8(buffer_u8[1]),
            F::from_canonical_u8(buffer_u8[2]),
            F::from_canonical_u8(buffer_u8[3])    
        ];
        trace_row.instruction.buffer_limbs = buffer_limbs;
        let input_u8: [u8; 4] = record.input.to_le_bytes();
        let input_limbs: [F; 4] = [
            F::from_canonical_u8(input_u8[0]), 
            F::from_canonical_u8(input_u8[1]),
            F::from_canonical_u8(input_u8[2]),
            F::from_canonical_u8(input_u8[3])    
        ];
        trace_row.instruction.input_limbs = input_limbs;
        let len_u8: [u8; 4] = record.len.to_le_bytes();
        let len_limbs: [F; 4] = [
            F::from_canonical_u8(len_u8[0]),
            F::from_canonical_u8(len_u8[1]),
            F::from_canonical_u8(len_u8[2]),
            F::from_canonical_u8(len_u8[3])
        ];
        trace_row.instruction.len_limbs = len_limbs; 
        
        for i in 0..(record.len/4) {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ZERO;
        }
        for i in (record.len/4)..34 {
            trace_row.sponge.is_padding_bytes[i as usize] = F::ONE;
        }

        println!("record.len = {}", record.len);

        // todo: think if it is fine to leave the other record.len..34 bits empty
        for i in 0..record.len {
            trace_row.sponge.preimage_buffer_bytes[i as usize] = F::from_canonical_u8(record.buffer_limbs[i as usize]);
            trace_row.sponge.input_bytes[i as usize] = F::from_canonical_u8(record.input_limbs[i as usize]);
            trace_row.sponge.postimage_buffer_bytes[i as usize] = F::from_canonical_u8(record.buffer_limbs[i as usize] ^ record.input_limbs[i as usize]);
            let b_val = record.input_limbs[i as usize] as u32; 
            let c_val = record.buffer_limbs[i as usize] as u32; 
            self.bitwise_lookup_chip.request_xor(b_val, c_val);
        }

        let mut timestamp = record.timestamp;

        // todo: think if the order matters here (maybe due to timestamp things), but this should be fine since it is matched with the one in preflight
        for t in 0..3 {
            mem_helper.fill(
                record.register_aux_cols[t].prev_timestamp,
                timestamp, 
                trace_row.mem_oc.register_aux_cols[t].as_mut() 
            );

            timestamp += 1;
        }
        
        for t in 0..34 {
            mem_helper.fill(
                record.buffer_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_read_aux_cols[t].as_mut() 
            );
            timestamp += 1;
        }

        for t in 0..34 {
            mem_helper.fill(
                record.input_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.input_bytes_read_aux_cols[t].as_mut() 
            );
            timestamp += 1;
        }

        for t in 0..34 {
            mem_helper.fill(
                record.buffer_write_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_write_aux_cols[t].as_mut() 
            );
            timestamp += 1;
        }        
    }
}