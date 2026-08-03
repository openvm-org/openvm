use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::*,
    system::memory::{offline_checker::MemoryReadAuxRecord, MemoryAuxColsFactory},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::U16_BITS;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_riscv_circuit::adapters::{
    byte_ptr_to_u16_ptr_value, ptr_bound_from_ptr, ptr_to_field_u16_limbs, rv64_bytes_to_u16_block,
    rv64_u16_block_to_bytes, try_rv64_bytes_to_u32,
};
use openvm_stark_backend::{
    p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix, p3_maybe_rayon::prelude::*,
};

use crate::{
    xorin::{columns::XorinVmCols, XorinVmChip, XorinVmFiller},
    KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS,
};

struct XorinTraceInput {
    from_pc: u32,
    timestamp: u32,
    rd_ptr: u32,
    rs1_ptr: u32,
    rs2_ptr: u32,
    buffer: u32,
    input: u32,
    len: u32,
    buffer_limbs: [u8; KECCAK_RATE_BYTES],
    input_limbs: [u8; KECCAK_RATE_BYTES],
    register_aux_cols: [MemoryReadAuxRecord; 3],
    input_read_aux_cols: [MemoryReadAuxRecord; KECCAK_RATE_MEM_OPS],
    buffer_read_aux_cols: [MemoryReadAuxRecord; KECCAK_RATE_MEM_OPS],
    buffer_write_prev_timestamps: [u32; KECCAK_RATE_MEM_OPS],
}

impl XorinVmFiller {
    fn fill_trace_input<F: PrimeField32>(
        &self,
        mem_helper: &MemoryAuxColsFactory<F>,
        input: &XorinTraceInput,
        row_slice: &mut [F],
    ) {
        row_slice.fill(F::ZERO);
        let trace_row: &mut XorinVmCols<F> = row_slice.borrow_mut();

        trace_row.instruction.pc = F::from_u32(input.from_pc);
        trace_row.instruction.is_enabled = F::ONE;
        trace_row.instruction.buffer_reg_ptr = F::from_u32(input.rd_ptr);
        trace_row.instruction.input_reg_ptr = F::from_u32(input.rs1_ptr);
        trace_row.instruction.len_reg_ptr = F::from_u32(input.rs2_ptr);
        trace_row.instruction.buffer_ptr_limbs = ptr_to_field_u16_limbs(input.buffer);
        trace_row.instruction.input_ptr_limbs = ptr_to_field_u16_limbs(input.input);
        trace_row.instruction.start_timestamp = F::from_u32(input.timestamp);

        for i in 0..(input.len as usize / MEMORY_BLOCK_BYTES) {
            trace_row.sponge.is_padding_bytes[i] = F::ZERO;
        }
        for i in (input.len as usize / MEMORY_BLOCK_BYTES)..(KECCAK_RATE_MEM_OPS) {
            trace_row.sponge.is_padding_bytes[i] = F::ONE;
        }

        let mut timestamp = input.timestamp;
        let input_len = input.len as usize;
        let num_reads = input_len / MEMORY_BLOCK_BYTES;

        for t in 0..3 {
            mem_helper.fill(
                input.register_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.register_aux_cols[t].as_mut(),
            );

            timestamp += 1;
        }

        for t in 0..num_reads {
            mem_helper.fill(
                input.buffer_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.buffer_bytes_read_aux_cols[t].as_mut(),
            );
            timestamp += 1;
        }

        for t in 0..num_reads {
            mem_helper.fill(
                input.input_read_aux_cols[t].prev_timestamp,
                timestamp,
                trace_row.mem_oc.input_bytes_read_aux_cols[t].as_mut(),
            );
            timestamp += 1;
        }

        for i in 0..input_len {
            trace_row.sponge.preimage_buffer_bytes[i] = F::from_u8(input.buffer_limbs[i]);
            trace_row.sponge.input_bytes[i] = F::from_u8(input.input_limbs[i]);
            trace_row.sponge.postimage_buffer_bytes[i] =
                F::from_u8(input.buffer_limbs[i] ^ input.input_limbs[i]);
            let b_val = input.buffer_limbs[i] as u32;
            let c_val = input.input_limbs[i] as u32;
            self.bitwise_lookup_chip.request_xor(b_val, c_val);
        }

        for t in 0..num_reads {
            mem_helper.fill(
                input.buffer_write_prev_timestamps[t],
                timestamp,
                &mut trace_row.mem_oc.buffer_bytes_write_base_aux[t],
            );
            timestamp += 1;
        }

        for ptr in [input.buffer, input.input] {
            self.range_checker_chip
                .add_count(ptr_bound_from_ptr(ptr, self.pointer_max_bits), U16_BITS);
        }
    }
}

/// Generates the XORIN trace directly from immutable preflight history.
pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &XorinVmChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(XorinOpcode::XORIN.global_opcode());
    let width = XorinVmCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let inputs = steps
        .par_iter()
        .map(|&step| replay_input(postflight, step, chip.inner.pointer_max_bits))
        .collect::<Result<Vec<_>, _>>()?;
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();
    trace
        .values
        .par_chunks_exact_mut(width)
        .zip(inputs.par_iter())
        .for_each(|(row, input)| chip.inner.fill_trace_input(&mem_helper, input, row));
    Ok(trace)
}

fn replay_input<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    pointer_max_bits: usize,
) -> Result<XorinTraceInput, PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
        || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
    {
        return Err(PostflightError::new(
            "XORIN instruction has invalid address spaces",
        ));
    }

    let from_pc = postflight.pc(step);
    let timestamp = postflight.timestamp(step);
    let rd_ptr = instruction.a.as_canonical_u32();
    let rs1_ptr = instruction.b.as_canonical_u32();
    let rs2_ptr = instruction.c.as_canonical_u32();
    for pointer in [rd_ptr, rs1_ptr, rs2_ptr] {
        if pointer & 1 != 0 {
            return Err(PostflightError::new(
                "XORIN register pointer must be two-byte aligned",
            ));
        }
    }

    let mut replay = postflight.replay(step);
    let register_reads = [
        replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(rd_ptr))?,
        replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(rs1_ptr))?,
        replay.read_u16(RV64_REGISTER_AS, byte_ptr_to_u16_ptr_value(rs2_ptr))?,
    ];
    let [Some(buffer), Some(input), Some(len)] = register_reads
        .each_ref()
        .map(|access| try_rv64_bytes_to_u32(rv64_u16_block_to_bytes(access.value)))
    else {
        return Err(PostflightError::new("XORIN register value exceeds 32 bits"));
    };
    let len_usize = len as usize;
    if len_usize > KECCAK_RATE_BYTES || !len_usize.is_multiple_of(MEMORY_BLOCK_BYTES) {
        return Err(PostflightError::new(
            "XORIN length must be an aligned rate-sized byte count",
        ));
    }
    let num_reads = len_usize / MEMORY_BLOCK_BYTES;
    if num_reads != 0 && (buffer & 1 != 0 || input & 1 != 0) {
        return Err(PostflightError::new(
            "XORIN memory pointer must be two-byte aligned",
        ));
    }
    let domain_end = if pointer_max_bits < 32 {
        1u64 << pointer_max_bits
    } else {
        1u64 << 32
    };
    if u64::from(buffer) >= domain_end
        || u64::from(input) >= domain_end
        || u64::from(buffer) + u64::from(len) > domain_end
        || u64::from(input) + u64::from(len) > domain_end
    {
        return Err(PostflightError::new(
            "XORIN memory range exceeds the pointer domain",
        ));
    }

    let mut buffer_limbs = [0u8; KECCAK_RATE_BYTES];
    let mut input_limbs = [0u8; KECCAK_RATE_BYTES];
    let mut buffer_read_aux_cols =
        std::array::from_fn(|_| MemoryReadAuxRecord { prev_timestamp: 0 });
    let mut input_read_aux_cols =
        std::array::from_fn(|_| MemoryReadAuxRecord { prev_timestamp: 0 });
    for index in 0..num_reads {
        let access = replay.read_u16(
            RV64_MEMORY_AS,
            byte_ptr_to_u16_ptr_value(buffer) + (index * BLOCK_FE_WIDTH) as u32,
        )?;
        buffer_limbs[index * MEMORY_BLOCK_BYTES..(index + 1) * MEMORY_BLOCK_BYTES]
            .copy_from_slice(&rv64_u16_block_to_bytes(access.value));
        buffer_read_aux_cols[index].prev_timestamp = access.previous_timestamp;
    }
    for index in 0..num_reads {
        let access = replay.read_u16(
            RV64_MEMORY_AS,
            byte_ptr_to_u16_ptr_value(input) + (index * BLOCK_FE_WIDTH) as u32,
        )?;
        input_limbs[index * MEMORY_BLOCK_BYTES..(index + 1) * MEMORY_BLOCK_BYTES]
            .copy_from_slice(&rv64_u16_block_to_bytes(access.value));
        input_read_aux_cols[index].prev_timestamp = access.previous_timestamp;
    }

    let mut buffer_write_prev_timestamps = [0; KECCAK_RATE_MEM_OPS];
    for (index, prev_timestamp) in buffer_write_prev_timestamps
        .iter_mut()
        .enumerate()
        .take(num_reads)
    {
        let mut output = [0u8; MEMORY_BLOCK_BYTES];
        for (byte, output_byte) in output.iter_mut().enumerate() {
            let offset = index * MEMORY_BLOCK_BYTES + byte;
            *output_byte = buffer_limbs[offset] ^ input_limbs[offset];
        }
        let access = replay.write_u16(
            RV64_MEMORY_AS,
            byte_ptr_to_u16_ptr_value(buffer) + (index * BLOCK_FE_WIDTH) as u32,
            rv64_bytes_to_u16_block(output),
        )?;
        *prev_timestamp = access.previous_timestamp;
    }
    let next_pc = from_pc
        .checked_add(DEFAULT_PC_STEP)
        .ok_or_else(|| PostflightError::new("XORIN program counter overflow"))?;
    replay.finish(next_pc)?;

    Ok(XorinTraceInput {
        from_pc,
        timestamp,
        rd_ptr,
        rs1_ptr,
        rs2_ptr,
        buffer,
        input,
        len,
        buffer_limbs,
        input_limbs,
        register_aux_cols: register_reads.map(|access| MemoryReadAuxRecord {
            prev_timestamp: access.previous_timestamp,
        }),
        input_read_aux_cols,
        buffer_read_aux_cols,
        buffer_write_prev_timestamps,
    })
}
