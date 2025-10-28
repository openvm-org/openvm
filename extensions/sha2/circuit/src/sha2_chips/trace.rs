use std::{
    borrow::{Borrow, BorrowMut},
    mem::transmute,
    slice::{from_raw_parts, from_raw_parts_mut},
};

use itertools::izip;
use ndarray::s;
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
use openvm_stark_backend::p3_field::PrimeField32;
use sha2::compress256;

use crate::{
    Sha256Config, Sha2BlockHasherConfig, Sha2ColsRef, Sha2ColsRefMut, Sha2Config,
    Sha2MainChipConfig, Sha2Variant, Sha2VmExecutor, Sha384Config, Sha512Config, SHA2_READ_SIZE,
    SHA2_REGISTER_READS, SHA2_WRITE_SIZE,
};

#[derive(Clone, Copy)]
pub struct Sha2Metadata {
    pub variant: Sha2Variant,
}

impl MultiRowMetadata for Sha2Metadata {
    #[inline(always)]
    fn get_num_rows(&self) -> usize {
        match self.variant {
            Sha2Variant::Sha256 => Sha256Config::ROWS_PER_BLOCK,
            Sha2Variant::Sha512 => Sha512Config::ROWS_PER_BLOCK,
            Sha2Variant::Sha384 => Sha384Config::ROWS_PER_BLOCK,
        }
    }
}

pub(crate) type Sha2RecordLayout = MultiRowLayout<Sha2Metadata>;

// #[derive(Clone, Copy, Debug, ColsRef)]
// #[config(Sha2Config)]
// struct test<T, C: Sha2Config> {
//     pub a: T,
//     pub b: [T; {STATE_BYTES}],
//     pub c: [u8; {STATE_BYTES}],
// }

// #[derive(Clone, Copy, Debug, ColsRef)]
// #[config(Sha2Config)]
// pub struct Sha2RecordMut<'a, C: Sha2Config> {
//     #[aligned_borrow]
//     // pub variant: Sha2Variant,
//     pub variant: u32,
//     #[aligned_borrow]
//     pub from_pc: u32,
//     #[aligned_borrow]
//     pub timestamp: u32,
//     #[aligned_borrow]
//     pub dst_reg_ptr: u32,
//     #[aligned_borrow]
//     pub state_reg_ptr: u32,
//     #[aligned_borrow]
//     pub input_reg_ptr: u32,
//     #[aligned_borrow]
//     pub dst_ptr: u32,
//     pub state_ptr: u32,
//     #[aligned_borrow]
//     pub input_ptr: u32,

//     pub state: [u8; { STATE_BYTES }],
//     pub input: [u8; { BLOCK_BYTES }],

//     #[aligned_borrow]
//     pub register_reads_aux: [MemoryReadAuxRecord; SHA2_REGISTER_READS],
//     #[aligned_borrow]
//     pub input_reads_aux: [MemoryReadAuxRecord; { BLOCK_READS }],
//     #[aligned_borrow]
//     pub state_reads_aux: [MemoryReadAuxRecord; { STATE_READS }],
//     #[aligned_borrow]
//     pub write_aux: [MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>; { DIGEST_WRITES }],
// }

/*
#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct Sha2RecordHeader {
    pub is_enabled: u32, // empty
    pub from_pc: u32,
    pub timestamp: u32,
    pub dst_reg_ptr: u32,
    pub state_reg_ptr: u32,
    pub input_reg_ptr: u32,
    // only the first element of each array is used, but we want to leave space for the rest
    // since in the columns struct, the ptrs are represented as limbs
    pub dst_ptr: [u32; RV32_REGISTER_NUM_LIMBS],
    pub state_ptr: [u32; RV32_REGISTER_NUM_LIMBS],
    pub input_ptr: [u32; RV32_REGISTER_NUM_LIMBS],

    pub register_reads_aux: [MemoryReadAuxRecord; SHA2_REGISTER_READS],
}

// Some fields in the record struct will not be filled in by the preflight executor,
// but we would like to leave space for them so that the record's memory
// layout matches the trace columns.
pub struct Sha2RecordMut<'a> {
    pub request_id: u32, // empty space
    pub input: &'a mut [u8],
    pub prev_state: &'a mut [u8],
    pub new_state: &'a mut [u8], // empty space

    pub inner: &'a mut Sha2RecordHeader,

    pub input_reads_aux: &'a mut [MemoryReadAuxRecord],
    pub state_reads_aux: &'a mut [MemoryReadAuxRecord],
    pub write_aux: &'a mut [MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>],
}
*/

/*
impl<'a> CustomBorrow<'a, Sha2RecordMut<'a>, Sha2RecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Sha2RecordLayout) -> Sha2RecordMut<'a> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits and
        //   constants are guaranteed <= self.len() by layout precondition

        let dims = Sha2PreComputeDims::new(layout.metadata.variant);

        let (request_id, rest) = unsafe { align_to_mut_at(self, 1) };
        let request_id: u32 = request_id[0];

        let (input, rest) = unsafe { rest.split_at_mut_unchecked(dims.input_size) };
        let (prev_state, rest) = unsafe { rest.split_at_mut_unchecked(dims.state_size) };
        let (new_state, rest) = unsafe { rest.split_at_mut_unchecked(dims.state_size) };

        let (header_slice, rest) =
            unsafe { rest.split_at_mut_unchecked(size_of::<Sha2RecordHeader>()) };
        let record_header: &mut Sha2RecordHeader = header_slice.borrow_mut();

        let (input_reads_aux, rest) = unsafe { align_to_mut_at(rest, dims.input_reads) };
        let (state_reads_aux, rest) = unsafe { align_to_mut_at(rest, dims.state_reads) };
        let (write_aux, _) = unsafe { align_to_mut_at(rest, dims.digest_writes) };

        Sha2RecordMut {
            variant,
            request_id,
            input,
            prev_state,
            new_state,
            inner: record_header,
            input_reads_aux,
            state_reads_aux,
            write_aux,
        }
    }

    unsafe fn extract_layout(&self) -> Sha2RecordLayout {
        let (variant, rest) = unsafe { align_to_mut_at(self, 1) };
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
        from_raw_parts_mut(items_rest.as_mut_ptr(), items_rest.len() + rest.len())
    };
    (items, rest)
}

impl SizedRecord<Sha2RecordLayout> for Sha2RecordMut<'_> {
    fn size(layout: &Sha2RecordLayout) -> usize {
        let header_size = size_of::<Sha2RecordHeader>();
        let dims = Sha2PreComputeDims::new(layout.metadata.variant);
        1 // variant
            + 1 // request_id
            + dims.input_size // input
            + dims.state_size // prev_state
            + dims.state_size // new_state
            + header_size
            + dims.input_reads * size_of::<MemoryReadAuxRecord>()
            + dims.state_reads * size_of::<MemoryReadAuxRecord>()
            + dims.digest_writes * size_of::<MemoryWriteBytesAuxRecord<SHA2_WRITE_SIZE>>()
    }

    fn alignment(_layout: &Sha2RecordLayout) -> usize {
        // TODO: is this correct?
        align_of::<Sha2RecordHeader>()
    }
}
*/

impl<'a> CustomBorrow<'a, Sha2ColsRefMut<'a, u32>, Sha2RecordLayout> for [u8] {
    fn custom_borrow(&'a mut self, layout: Sha2RecordLayout) -> Sha2ColsRefMut<'a, u32> {
        // SAFETY:
        // - Caller guarantees through the layout that self has sufficient length for all splits and
        //   constants are guaranteed <= self.len() by layout precondition

        let slice: &'a mut [u32] = unsafe { transmute(self) };
        match layout.metadata.variant {
            Sha2Variant::Sha256 => Sha2ColsRefMut::from::<Sha256Config>(slice),
            Sha2Variant::Sha512 => Sha2ColsRefMut::from::<Sha512Config>(slice),
            Sha2Variant::Sha384 => Sha2ColsRefMut::from::<Sha384Config>(slice),
        }
    }

    unsafe fn extract_layout(&self) -> Sha2RecordLayout {
        // we will pack the variant into the first element of the slice (i.e cols.block.request_id
        // in the columns struct)
        let (variant, _) = unsafe { align_to_at(&self, 1) };
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
        from_raw_parts_mut(items_rest.as_mut_ptr(), items_rest.len() + rest.len())
    };
    (items, rest)
}

unsafe fn align_to_at<T>(slice: &[u8], offset: usize) -> (&[T], &[u8]) {
    let (_, items, rest) = unsafe { slice.align_to::<T>() };
    let (items, items_rest) = unsafe { items.split_at_unchecked(offset) };
    let rest = unsafe {
        let items_rest: &[u8] = transmute(items_rest);
        from_raw_parts(items_rest.as_ptr(), items_rest.len() + rest.len())
    };
    (items, rest)
}

impl SizedRecord<Sha2RecordLayout> for Sha2ColsRefMut<'_, u32> {
    fn size(layout: &Sha2RecordLayout) -> usize {
        match layout.metadata.variant {
            Sha2Variant::Sha256 => {
                Sha2ColsRefMut::<u32>::width::<Sha256Config>() * size_of::<u32>()
            }
            Sha2Variant::Sha512 => {
                Sha2ColsRefMut::<u32>::width::<Sha512Config>() * size_of::<u32>()
            }
            Sha2Variant::Sha384 => {
                Sha2ColsRefMut::<u32>::width::<Sha384Config>() * size_of::<u32>()
            }
        }
    }

    fn alignment(_layout: &Sha2RecordLayout) -> usize {
        // TODO: is this correct?
        align_of::<u32>()
    }
}

struct Sha2PreComputeDims {
    state_size: usize,
    input_size: usize,
    input_reads: usize,
    state_reads: usize,
    digest_writes: usize,
}

impl Sha2PreComputeDims {
    fn new(variant: Sha2Variant) -> Self {
        match variant {
            Sha2Variant::Sha256 => Self {
                state_size: Sha256Config::STATE_BYTES,
                input_size: Sha256Config::BLOCK_BYTES,
                input_reads: Sha256Config::BLOCK_READS,
                state_reads: Sha256Config::STATE_READS,
                digest_writes: Sha256Config::DIGEST_WRITES,
            },
            Sha2Variant::Sha512 => Self {
                state_size: Sha512Config::STATE_BYTES,
                input_size: Sha512Config::BLOCK_BYTES,
                input_reads: Sha512Config::BLOCK_READS,
                state_reads: Sha512Config::STATE_READS,
                digest_writes: Sha512Config::DIGEST_WRITES,
            },
            Sha2Variant::Sha384 => Self {
                state_size: Sha384Config::STATE_BYTES,
                input_size: Sha384Config::BLOCK_BYTES,
                input_reads: Sha384Config::BLOCK_READS,
                state_reads: Sha384Config::STATE_READS,
                digest_writes: Sha384Config::DIGEST_WRITES,
            },
        }
    }
}

impl<F, RA, C: Sha2Config> PreflightExecutor<F, RA> for Sha2VmExecutor<C>
where
    F: PrimeField32,
    // for<'buf> RA: RecordArena<'buf, Sha2RecordLayout, Sha2RecordMut<'buf>>,
    for<'buf> RA: RecordArena<'buf, Sha2RecordLayout, Sha2ColsRefMut<'buf, u32>>,
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

        let mut record = state.ctx.alloc(Sha2RecordLayout::new(Sha2Metadata {
            variant: C::VARIANT,
        }));

        record.instruction.from_state.pc = *state.pc;
        record.instruction.from_state.timestamp = state.memory.timestamp();
        *record.instruction.dst_reg_ptr = a.as_canonical_u32();
        *record.instruction.state_reg_ptr = b.as_canonical_u32();
        *record.instruction.input_reg_ptr = c.as_canonical_u32();

        let rest = record.mem.register_aux.as_slice_mut().unwrap();
        let (dst_reg_aux, rest) = unsafe { rest.split_at_mut_unchecked(1) };
        let (state_reg_aux, rest) = unsafe { rest.split_at_mut_unchecked(1) };
        let (input_reg_aux, _) = unsafe { rest.split_at_mut_unchecked(1) };
        let mut dst_reg_aux = dst_reg_aux[0];
        let mut state_reg_aux = state_reg_aux[0];
        let mut input_reg_aux = input_reg_aux[0];

        for (ptr, prev_timestamp, data) in izip!(
            [
                record.instruction.dst_reg_ptr,
                record.instruction.state_reg_ptr,
                record.instruction.input_reg_ptr
            ],
            [
                &mut dst_reg_aux.base.prev_timestamp,
                &mut state_reg_aux.base.prev_timestamp,
                &mut input_reg_aux.base.prev_timestamp
            ],
            [
                &mut record.instruction.dst_ptr_limbs,
                &mut record.instruction.state_ptr_limbs,
                &mut record.instruction.input_ptr_limbs
            ],
        ) {
            let read = tracing_read::<SHA2_READ_SIZE>(
                state.memory,
                RV32_REGISTER_AS,
                *ptr,
                prev_timestamp,
            );
            data.iter_mut()
                .zip(read)
                .for_each(|(dst_ptr_limb, read_val)| {
                    *dst_ptr_limb = read_val.into();
                });
        }

        let dst_ptr: u32 = u32::from_le_bytes(
            record
                .instruction
                .dst_ptr_limbs
                .iter()
                .map(|x| *x as u8)
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
        let state_ptr: u32 = u32::from_le_bytes(
            record
                .instruction
                .state_ptr_limbs
                .iter()
                .map(|x| *x as u8)
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );
        let input_ptr: u32 = u32::from_le_bytes(
            record
                .instruction
                .input_ptr_limbs
                .iter()
                .map(|x| *x as u8)
                .collect::<Vec<u8>>()
                .try_into()
                .unwrap(),
        );

        debug_assert!(dst_ptr as usize + C::DIGEST_BYTES <= (1 << self.pointer_max_bits));
        debug_assert!(state_ptr as usize + C::STATE_BYTES <= (1 << self.pointer_max_bits));
        debug_assert!(input_ptr as usize + C::BLOCK_BYTES <= (1 << self.pointer_max_bits));

        for idx in 0..C::BLOCK_READS {
            let read = tracing_read::<SHA2_READ_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                input_ptr + (idx * SHA2_READ_SIZE) as u32,
                &mut record.mem.input_reads[idx].base.prev_timestamp,
            );
            record
                .block
                .message_bytes
                .slice_mut(s![idx * SHA2_READ_SIZE..(idx + 1) * SHA2_READ_SIZE])
                .iter_mut()
                .zip(read.iter())
                .for_each(|(dst, src)| *dst = (*src).into());
        }

        for idx in 0..C::STATE_READS {
            let read = tracing_read::<SHA2_READ_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                state_ptr + (idx * SHA2_READ_SIZE) as u32,
                &mut record.mem.state_reads[idx].base.prev_timestamp,
            );
            record
                .block
                .prev_state
                .slice_mut(s![idx * SHA2_READ_SIZE..(idx + 1) * SHA2_READ_SIZE])
                .iter_mut()
                .zip(read.iter())
                .for_each(|(dst, src)| *dst = (*src).into());
        }

        let mut curr_state = record
            .block
            .prev_state
            .iter()
            .map(|x| *x as u8)
            .collect::<Vec<u8>>();
        C::compress(
            &mut curr_state,
            record
                .block
                .message_bytes
                .iter()
                .map(|x| *x as u8)
                .collect::<Vec<u8>>()
                .as_slice(),
        );

        for idx in 0..C::DIGEST_WRITES {
            let mut prev_data: [u8; SHA2_WRITE_SIZE] = [0; SHA2_WRITE_SIZE];
            tracing_write::<SHA2_WRITE_SIZE>(
                state.memory,
                RV32_MEMORY_AS,
                dst_ptr + (idx * SHA2_WRITE_SIZE) as u32,
                curr_state[idx * SHA2_WRITE_SIZE..(idx + 1) * SHA2_WRITE_SIZE]
                    .try_into()
                    .unwrap(),
                &mut record.mem.write_aux[idx].base.prev_timestamp,
                &mut prev_data,
            );
            record.mem.write_aux[idx]
                .prev_data
                .iter_mut()
                .zip(prev_data.iter())
                .for_each(|(dst, src)| *dst = (*src).into());
        }

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}
