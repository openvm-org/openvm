use std::{
    borrow::{Borrow, BorrowMut},
    mem::transmute,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use openvm_circuit::{
    arch::{
        CustomBorrow, ExecutionError, MultiRowLayout, MultiRowMetadata, PreflightExecutor,
        RecordArena, SizedRecord, VmStateMut,
    },
    system::memory::{
        offline_checker::{MemoryReadAuxRecord, MemoryWriteBytesAuxRecord},
        online::TracingMemory,
    },
};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{
    instruction::Instruction,
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_rv32im_circuit::adapters::{tracing_read, tracing_write};
use openvm_sha2_air::{
    Sha256Config, Sha2BlockHasherSubairConfig, Sha2Variant, Sha384Config, Sha512Config,
};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    Sha2Config, Sha2MainChipConfig, Sha2VmExecutor, SHA2_READ_SIZE, SHA2_REGISTER_READS,
    SHA2_WRITE_SIZE,
};

#[derive(Clone, Copy)]
pub struct Sha2Metadata {
    pub variant: Sha2Variant,
}

impl MultiRowMetadata for Sha2Metadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        // The size of the record arena will be height * Sha2MainAir::width() * num_rows.
        // We will not use the record arena's buffer for either chip's trace, so we just
        // need to ensure that the record arena is large enough to store all the records.
        // The size of Sha2RecordMut (in bytes) is less than Sha2MainAir::width() * size_of::<F>(),
        // for all SHA-2 variants. Therefore, we can set num_rows = 1.
        1
    }
}

pub(crate) type Sha2RecordLayout = MultiRowLayout<Sha2Metadata>;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Sha2RecordHeader {
    pub variant: Sha2Variant,
    pub from_pc: u32,
    pub timestamp: u32,
    pub dst_reg_ptr: u32,
    pub state_reg_ptr: u32,
    pub input_reg_ptr: u32,
    pub dst_ptr: u32,
    pub state_ptr: u32,
    pub input_ptr: u32,

    pub register_reads_aux: [MemoryReadAuxRecord; SHA2_REGISTER_READS],
}

pub struct Sha2RecordMut<'a> {
    pub inner: &'a mut Sha2RecordHeader,

    pub message_bytes: &'a mut [u8],
    pub prev_state: &'a mut [u8], // little-endian words
    pub new_state: &'a mut [u8],  // little-endian words

    pub input_reads_aux: &'a mut [MemoryReadAuxRecord],
    pub state_reads_aux: &'a mut [MemoryReadAuxRecord],
    pub write_aux: &'a mut [MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>],
}

impl<'a> CustomBorrow<'a, Sha2RecordMut<'a>, Sha2RecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Sha2RecordLayout) -> Sha2RecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits and
        //   constants are guaranteed <= self.len() by layout precondition

        let (header_slice, rest) =
            unsafe { self.split_at_mut_unchecked(size_of::<Sha2RecordHeader>()) };
        let record_header: &mut Sha2RecordHeader = header_slice.borrow_mut();

        let dims = Sha2PreComputeDims::new(layout.metadata.variant);

        let (message_bytes, rest) = unsafe { rest.split_at_mut_unchecked(dims.input_size) };
        let (prev_state, rest) = unsafe { rest.split_at_mut_unchecked(dims.state_size) };
        let (new_state, rest) = unsafe { rest.split_at_mut_unchecked(dims.state_size) };

        let (input_reads_aux, rest) = unsafe { align_to_mut_at(rest, dims.input_reads) };
        let (state_reads_aux, rest) = unsafe { align_to_mut_at(rest, dims.state_reads) };
        let (write_aux, _) = unsafe { align_to_mut_at(rest, dims.state_writes) };

        Sha2RecordMut {
            inner: record_header,
            message_bytes,
            prev_state,
            new_state,
            input_reads_aux,
            state_reads_aux,
            write_aux,
        }
    }

    unsafe fn extract_layout(&self) -> Sha2RecordLayout {
        let (variant, _) = unsafe { align_to_at(self, 1) };
        let variant = variant[0];
        Sha2RecordLayout {
            metadata: Sha2Metadata { variant },
        }
    }
}

unsafe fn align_to_mut_at<T>(slice: &mut [u8], offset: usize) -> (&mut [T], &mut [u8]) {
    let (_, items, rest) = unsafe { slice.align_to_mut::<T>() };
    let (items, items_rest) = unsafe { items.split_at_mut_unchecked(offset) };
    let rest = unsafe {
        let items_rest: &mut [u8] = transmute(items_rest);
        from_raw_parts_mut(
            items_rest.as_mut_ptr(),
            items_rest.len() * size_of::<T>() + rest.len(),
        )
    };
    (items, rest)
}

unsafe fn align_to_at<T>(slice: &[u8], offset: usize) -> (&[T], &[u8]) {
    let (_, items, rest) = unsafe { slice.align_to::<T>() };
    let (items, items_rest) = unsafe { items.split_at_unchecked(offset) };
    let rest = unsafe {
        let items_rest: &[u8] = transmute(items_rest);
        from_raw_parts(
            items_rest.as_ptr(),
            items_rest.len() * size_of::<T>() + rest.len(),
        )
    };
    (items, rest)
}

impl SizedRecord<Sha2RecordLayout> for Sha2RecordMut<'_> {
    fn size(layout: &Sha2RecordLayout) -> usize {
        let header_size = size_of::<Sha2RecordHeader>();
        let dims = Sha2PreComputeDims::new(layout.metadata.variant);
        let mut total_len = header_size
            + dims.input_size  // input
            + dims.state_size  // prev_state
            + dims.state_size; // new_state

        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += dims.input_reads * size_of::<MemoryReadAuxRecord>();

        total_len = total_len.next_multiple_of(align_of::<MemoryReadAuxRecord>());
        total_len += dims.state_reads * size_of::<MemoryReadAuxRecord>();

        total_len =
            total_len.next_multiple_of(align_of::<MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>>());
        total_len += dims.state_writes * size_of::<MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>>();

        total_len
    }

    fn alignment(_layout: &Sha2RecordLayout) -> usize {
        // TODO: is this correct?
        align_of::<Sha2RecordHeader>()
    }
}

// This is needed in CustomBorrow trait to convert the Sha2Variant that we read from the buffer
// into appropriate dimensions for the record.
struct Sha2PreComputeDims {
    state_size: usize,
    input_size: usize,
    input_reads: usize,
    state_reads: usize,
    state_writes: usize,
}

impl Sha2PreComputeDims {
    fn new(variant: Sha2Variant) -> Self {
        match variant {
            Sha2Variant::Sha256 => Self {
                state_size: Sha256Config::STATE_BYTES,
                input_size: Sha256Config::BLOCK_BYTES,
                input_reads: Sha256Config::BLOCK_READS,
                state_reads: Sha256Config::STATE_READS,
                state_writes: Sha256Config::STATE_WRITES,
            },
            Sha2Variant::Sha512 => Self {
                state_size: Sha512Config::STATE_BYTES,
                input_size: Sha512Config::BLOCK_BYTES,
                input_reads: Sha512Config::BLOCK_READS,
                state_reads: Sha512Config::STATE_READS,
                state_writes: Sha512Config::STATE_WRITES,
            },
            Sha2Variant::Sha384 => Self {
                state_size: Sha384Config::STATE_BYTES,
                input_size: Sha384Config::BLOCK_BYTES,
                input_reads: Sha384Config::BLOCK_READS,
                state_reads: Sha384Config::STATE_READS,
                state_writes: Sha384Config::STATE_WRITES,
            },
        }
    }
}

impl<F, RA, C: Sha2Config> PreflightExecutor<F, RA> for Sha2VmExecutor<C>
where
    F: PrimeField32,
    // for<'buf> RA: RecordArena<'buf, Sha2RecordLayout, Sha2RecordMut<'buf>>,
    for<'buf> RA: RecordArena<'buf, Sha2RecordLayout, Sha2RecordMut<'buf>>,
{
    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", C::OPCODE)
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
        debug_assert_eq!(opcode, C::OPCODE.global_opcode());
        debug_assert_eq!(d.as_canonical_u32(), RV32_REGISTER_AS);
        debug_assert_eq!(e.as_canonical_u32(), RV32_MEMORY_AS);

        let record = state.ctx.alloc(Sha2RecordLayout::new(Sha2Metadata {
            variant: C::VARIANT,
        }));

        record.inner.variant = C::VARIANT;
        record.inner.from_pc = *state.pc;
        record.inner.timestamp = state.memory.timestamp();
        record.inner.dst_reg_ptr = a.as_canonical_u32();
        record.inner.state_reg_ptr = b.as_canonical_u32();
        record.inner.input_reg_ptr = c.as_canonical_u32();

        record.inner.dst_ptr = u32::from_le_bytes(tracing_read::<SHA2_READ_SIZE>(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.dst_reg_ptr,
            &mut record.inner.register_reads_aux[0].prev_timestamp,
        ));
        record.inner.state_ptr = u32::from_le_bytes(tracing_read::<SHA2_READ_SIZE>(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.state_reg_ptr,
            &mut record.inner.register_reads_aux[1].prev_timestamp,
        ));
        record.inner.input_ptr = u32::from_le_bytes(tracing_read::<SHA2_READ_SIZE>(
            state.memory,
            RV32_REGISTER_AS,
            record.inner.input_reg_ptr,
            &mut record.inner.register_reads_aux[2].prev_timestamp,
        ));

        debug_assert!(
            record.inner.dst_ptr as usize + C::STATE_BYTES <= (1 << self.pointer_max_bits)
        );
        debug_assert!(
            record.inner.state_ptr as usize + C::STATE_BYTES <= (1 << self.pointer_max_bits)
        );
        debug_assert!(
            record.inner.input_ptr as usize + C::BLOCK_BYTES <= (1 << self.pointer_max_bits)
        );

        for idx in 0..C::BLOCK_READS {
            let read = tracing_read::<SHA2_READ_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.input_ptr + (idx * SHA2_READ_SIZE) as u32,
                &mut record.input_reads_aux[idx].prev_timestamp,
            );
            record.message_bytes[idx * SHA2_READ_SIZE..(idx + 1) * SHA2_READ_SIZE]
                .copy_from_slice(&read);
        }

        for idx in 0..C::STATE_READS {
            let read = tracing_read::<SHA2_READ_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.state_ptr + (idx * SHA2_READ_SIZE) as u32,
                &mut record.state_reads_aux[idx].prev_timestamp,
            );
            record.prev_state[idx * SHA2_READ_SIZE..(idx + 1) * SHA2_READ_SIZE]
                .copy_from_slice(&read);
        }

        record.new_state.copy_from_slice(record.prev_state);
        C::compress(record.new_state, record.message_bytes);

        for idx in 0..C::STATE_WRITES {
            tracing_write::<SHA2_WRITE_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                record.inner.dst_ptr + (idx * SHA2_WRITE_SIZE) as u32,
                record.new_state[idx * SHA2_WRITE_SIZE..(idx + 1) * SHA2_WRITE_SIZE]
                    .try_into()
                    .unwrap(),
                &mut record.write_aux[idx].prev_timestamp,
                &mut record.write_aux[idx].prev_data,
            );
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}
