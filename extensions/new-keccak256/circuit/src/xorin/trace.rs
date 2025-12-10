use std::{
    borrow::{Borrow, BorrowMut}, fmt::format, mem::{align_of, size_of}
};

use openvm_circuit::{arch::*, system::memory::{offline_checker::MemoryReadAuxCols, online::TracingMemory}};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::instruction::Instruction;
use openvm_new_keccak256_transpiler::Rv32NewKeccakOpcode;
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_stark_backend::p3_field::PrimeField32;
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteAuxRecord;
use openvm_circuit::system::memory::offline_checker::MemoryWriteBytesAuxRecord;
use crate::xorin::trace::instructions::riscv::RV32_REGISTER_AS;
use crate::xorin::trace::instructions::riscv::RV32_MEMORY_AS;

use crate::xorin::XorinVmExecutor;

#[derive(Clone, Copy)]
pub struct XorinVmMetadata {
}

impl MultiRowMetadata for XorinVmMetadata {
    fn get_num_rows(&self) -> usize {
        0
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

// Custom borrowing to split the buffer into a fixed `KeccakVmRecord` header
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
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            .. 
        } = instruction;

        // Reading the length first without tracing to allocate a record of correct size
        let guest_mem = state.memory.data();
        let len = u32::from_le_bytes(unsafe {
            guest_mem.read::<u8, 4>(1, c.as_canonical_u32()) 
        }) as usize;
        // Safety: length has to be multiple of 4
        // This is enforced by how the guest program calls the xorin opcode
        // Xorin opcode is only called through the keccak update guest program
        debug_assert!(len % 4 == 0);
        let num_reads = len.div_ceil(4);
        let record = state
            .ctx 
            .alloc(XorinVmRecordLayout::new(XorinVmMetadata { len }));
        
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

        let result = [0u8; 136];

        // execute xorin
        for idx in 0..4*num_reads {
            result[idx] = record.inner.buffer_limbs[idx] ^ record.inner.input_limbs[idx];
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

        Ok(())
    }
}
