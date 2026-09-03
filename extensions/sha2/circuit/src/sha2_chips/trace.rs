use openvm_circuit::arch::{Postflight, PostflightError, PostflightStep};
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{MEMORY_AS, NUM_REGISTERS, REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::u16_block_to_bytes;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{Sha2Config, SHA2_READ_SIZE, SHA2_REGISTER_READS, SHA2_WRITE_SIZE};

pub(crate) struct Sha2ReplayRow {
    pub from_pc: u32,
    pub timestamp: u32,
    pub dst_reg_ptr: u32,
    pub state_reg_ptr: u32,
    pub input_reg_ptr: u32,
    pub dst_ptr: u32,
    pub state_ptr: u32,
    pub input_ptr: u32,
    pub register_prev_timestamps: [u32; SHA2_REGISTER_READS],
    pub message_bytes: Vec<u8>,
    pub prev_state: Vec<u8>,
    pub new_state: Vec<u8>,
    pub input_prev_timestamps: Vec<u32>,
    pub state_prev_timestamps: Vec<u32>,
    pub write_prev_timestamps: Vec<u32>,
    pub write_prev_data: Vec<[u8; SHA2_WRITE_SIZE]>,
}

pub(crate) fn replay_sha2_from_postflight<F, C>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
) -> Result<Sha2ReplayRow, PostflightError>
where
    F: PrimeField32,
    C: Sha2Config,
{
    let instruction = postflight.instruction(step);
    if instruction.opcode != C::OPCODE.global_opcode()
        || instruction.d.as_u32() != REGISTER_AS
        || instruction.e.as_u32() != MEMORY_AS
        || instruction.f.as_u32() != 0
        || instruction.g.as_u32() != 0
    {
        return Err(PostflightError::new(
            "SHA-2 instruction has invalid opcode or address spaces",
        ));
    }

    let register_ptrs = [
        instruction.a.as_u32(),
        instruction.b.as_u32(),
        instruction.c.as_u32(),
    ];
    if register_ptrs.iter().any(|&ptr| {
        ptr as usize >= NUM_REGISTERS * REGISTER_NUM_LIMBS
            || !(ptr as usize).is_multiple_of(REGISTER_NUM_LIMBS)
    }) {
        return Err(PostflightError::new(
            "SHA-2 instruction has a non-canonical register pointer",
        ));
    }

    let mut replay = postflight.replay(step);
    let mut pointer_values = [0u32; SHA2_REGISTER_READS];
    let mut register_prev_timestamps = [0u32; SHA2_REGISTER_READS];
    for (index, &register_ptr) in register_ptrs.iter().enumerate() {
        let access = replay.read_u16(REGISTER_AS, register_ptr >> 1)?;
        let pointer_bytes = u16_block_to_bytes(access.value);
        pointer_values[index] = u32::try_from(u64::from_le_bytes(pointer_bytes))
            .map_err(|_| PostflightError::new("SHA-2 memory pointer has nonzero upper 32 bits"))?;
        register_prev_timestamps[index] = access.previous_timestamp;
    }
    let [dst_ptr, state_ptr, input_ptr] = pointer_values;

    if pointer_max_bits > 32 {
        return Err(PostflightError::new("SHA-2 pointer bit width exceeds u32"));
    }
    let limit = 1u64 << pointer_max_bits;
    for (pointer, bytes) in [
        (dst_ptr, C::STATE_BYTES),
        (state_ptr, C::STATE_BYTES),
        (input_ptr, C::BLOCK_BYTES),
    ] {
        if !(pointer as usize).is_multiple_of(SHA2_READ_SIZE) {
            return Err(PostflightError::new(
                "SHA-2 memory pointer is not eight-byte aligned",
            ));
        }
        if u64::from(pointer)
            .checked_add(bytes as u64)
            .is_none_or(|end| end > limit)
        {
            return Err(PostflightError::new(
                "SHA-2 memory range exceeds the configured pointer domain",
            ));
        }
    }

    let mut message_bytes = Vec::with_capacity(C::BLOCK_BYTES);
    let mut input_prev_timestamps = Vec::with_capacity(C::BLOCK_READS);
    for index in 0..C::BLOCK_READS {
        let byte_ptr = input_ptr
            .checked_add((index * SHA2_READ_SIZE) as u32)
            .ok_or_else(|| PostflightError::new("SHA-2 input pointer overflow"))?;
        let access = replay.read_u16(MEMORY_AS, byte_ptr >> 1)?;
        message_bytes.extend_from_slice(&u16_block_to_bytes(access.value));
        input_prev_timestamps.push(access.previous_timestamp);
    }

    let mut prev_state = Vec::with_capacity(C::STATE_BYTES);
    let mut state_prev_timestamps = Vec::with_capacity(C::STATE_READS);
    for index in 0..C::STATE_READS {
        let byte_ptr = state_ptr
            .checked_add((index * SHA2_READ_SIZE) as u32)
            .ok_or_else(|| PostflightError::new("SHA-2 state pointer overflow"))?;
        let access = replay.read_u16(MEMORY_AS, byte_ptr >> 1)?;
        prev_state.extend_from_slice(&u16_block_to_bytes(access.value));
        state_prev_timestamps.push(access.previous_timestamp);
    }

    let mut new_state = Vec::with_capacity(C::STATE_BYTES);
    let mut write_prev_timestamps = Vec::with_capacity(C::STATE_WRITES);
    let mut write_prev_data = Vec::with_capacity(C::STATE_WRITES);
    for index in 0..C::STATE_WRITES {
        let byte_ptr = dst_ptr
            .checked_add((index * SHA2_WRITE_SIZE) as u32)
            .ok_or_else(|| PostflightError::new("SHA-2 destination pointer overflow"))?;
        let access = replay.write_observed_u16(MEMORY_AS, byte_ptr >> 1)?;
        new_state.extend_from_slice(&u16_block_to_bytes(access.value));
        write_prev_timestamps.push(access.previous_timestamp);
        write_prev_data.push(u16_block_to_bytes(access.previous_value));
    }

    let from_pc = postflight.pc(step);
    replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;
    Ok(Sha2ReplayRow {
        from_pc,
        timestamp: postflight.timestamp(step),
        dst_reg_ptr: register_ptrs[0],
        state_reg_ptr: register_ptrs[1],
        input_reg_ptr: register_ptrs[2],
        dst_ptr,
        state_ptr,
        input_ptr,
        register_prev_timestamps,
        message_bytes,
        prev_state,
        new_state,
        input_prev_timestamps,
        state_prev_timestamps,
        write_prev_timestamps,
        write_prev_data,
    })
}
