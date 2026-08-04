use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::*, system::memory::MemoryAuxColsFactory, utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_riscv_circuit::adapters::{
    byte_ptr_limbs_to_cell_ptr_limbs_value, byte_ptr_to_u16_ptr_value, compute_block_add_carries,
    compute_pointer_carries, ptr_to_field_u16_limbs, rv64_bytes_to_u16_block,
    rv64_u16_block_to_bytes, try_rv64_bytes_to_u32, u32_to_ptr_limbs,
};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use crate::{
    xorin::{columns::XorinVmCols, XorinVmChip, XorinVmFiller},
    KECCAK_RATE_BYTES, KECCAK_RATE_MEM_OPS,
};

impl XorinVmFiller {
    fn replay_and_fill_trace_row<F: PrimeField32>(
        &self,
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        row_slice: &mut [F],
    ) -> Result<(), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS
            || instruction.e.as_canonical_u32() != RV64_MEMORY_AS
        {
            return Err(PostflightError::new(
                "XORIN instruction has invalid address spaces",
            ));
        }

        let from_pc = postflight.pc(step);
        let start_timestamp = postflight.timestamp(step);
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
        let domain_end = if self.pointer_max_bits < 32 {
            1u64 << self.pointer_max_bits
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
        let mut buffer_read_prev_timestamps = [0; KECCAK_RATE_MEM_OPS];
        let mut input_read_prev_timestamps = [0; KECCAK_RATE_MEM_OPS];
        for index in 0..num_reads {
            let access = replay.read_u16(
                RV64_MEMORY_AS,
                byte_ptr_to_u16_ptr_value(buffer) + (index * BLOCK_FE_WIDTH) as u32,
            )?;
            buffer_limbs[index * MEMORY_BLOCK_BYTES..(index + 1) * MEMORY_BLOCK_BYTES]
                .copy_from_slice(&rv64_u16_block_to_bytes(access.value));
            buffer_read_prev_timestamps[index] = access.previous_timestamp;
        }
        for index in 0..num_reads {
            let access = replay.read_u16(
                RV64_MEMORY_AS,
                byte_ptr_to_u16_ptr_value(input) + (index * BLOCK_FE_WIDTH) as u32,
            )?;
            input_limbs[index * MEMORY_BLOCK_BYTES..(index + 1) * MEMORY_BLOCK_BYTES]
                .copy_from_slice(&rv64_u16_block_to_bytes(access.value));
            input_read_prev_timestamps[index] = access.previous_timestamp;
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

        row_slice.fill(F::ZERO);
        let trace_row: &mut XorinVmCols<F> = row_slice.borrow_mut();

        trace_row.instruction.pc = F::from_u32(from_pc);
        trace_row.instruction.is_enabled = F::ONE;
        trace_row.instruction.buffer_reg_ptr = F::from_u32(rd_ptr);
        trace_row.instruction.input_reg_ptr = F::from_u32(rs1_ptr);
        trace_row.instruction.len_reg_ptr = F::from_u32(rs2_ptr);
        trace_row.instruction.buffer_ptr_limbs = ptr_to_field_u16_limbs(buffer);
        trace_row.instruction.input_ptr_limbs = ptr_to_field_u16_limbs(input);
        trace_row.instruction.start_timestamp = F::from_u32(start_timestamp);

        for flag in &mut trace_row.sponge.is_padding_bytes[..num_reads] {
            *flag = F::ZERO;
        }
        for flag in &mut trace_row.sponge.is_padding_bytes[num_reads..] {
            *flag = F::ONE;
        }

        let mut timestamp = start_timestamp;
        for (access, aux) in register_reads
            .iter()
            .zip(&mut trace_row.mem_oc.register_aux_cols)
        {
            mem_helper.fill(access.previous_timestamp, timestamp, aux.as_mut());
            timestamp += 1;
        }
        for (&previous_timestamp, aux) in buffer_read_prev_timestamps[..num_reads]
            .iter()
            .zip(&mut trace_row.mem_oc.buffer_bytes_read_aux_cols)
        {
            mem_helper.fill(previous_timestamp, timestamp, aux.as_mut());
            timestamp += 1;
        }
        for (&previous_timestamp, aux) in input_read_prev_timestamps[..num_reads]
            .iter()
            .zip(&mut trace_row.mem_oc.input_bytes_read_aux_cols)
        {
            mem_helper.fill(previous_timestamp, timestamp, aux.as_mut());
            timestamp += 1;
        }

        for i in 0..len_usize {
            trace_row.sponge.preimage_buffer_bytes[i] = F::from_u8(buffer_limbs[i]);
            trace_row.sponge.input_bytes[i] = F::from_u8(input_limbs[i]);
            trace_row.sponge.postimage_buffer_bytes[i] =
                F::from_u8(buffer_limbs[i] ^ input_limbs[i]);
            self.bitwise_lookup_chip
                .request_xor(buffer_limbs[i] as u32, input_limbs[i] as u32);
        }

        for (&previous_timestamp, aux) in buffer_write_prev_timestamps[..num_reads]
            .iter()
            .zip(&mut trace_row.mem_oc.buffer_bytes_write_base_aux)
        {
            mem_helper.fill(previous_timestamp, timestamp, aux);
            timestamp += 1;
        }

        // Byte -> cell pointer conversion carries and per-block cell-offset carries, plus matching
        // range-check counts.
        //
        // The AIR gates the per-block cell-offset add by `is_enabled` (degree 1) rather than the
        // per-block `should_read`/`should_write` (degree 2) to stay within the max constraint
        // degree. So add carries (and their range checks) are computed for *every* block, padding
        // or not, matching the AIR's `is_enabled`-gated `eval_add_const_u16_limbs` for all blocks.
        let cell_stride = (MEMORY_BLOCK_BYTES / U16_CELL_SIZE) as u32;
        let (buffer_conv, buffer_add) = compute_pointer_carries(
            &self.range_checker_chip,
            buffer,
            KECCAK_RATE_MEM_OPS,
            cell_stride,
            self.pointer_max_bits,
        );
        trace_row.mem_oc.buffer_cell_carry = F::from_u32(buffer_conv);
        for (col, &add_carry) in trace_row
            .mem_oc
            .buffer_read_add_carry
            .iter_mut()
            .zip(buffer_add.iter())
        {
            *col = F::from_u32(add_carry);
        }
        let (input_conv, input_add) = compute_pointer_carries(
            &self.range_checker_chip,
            input,
            KECCAK_RATE_MEM_OPS,
            cell_stride,
            self.pointer_max_bits,
        );
        trace_row.mem_oc.input_cell_carry = F::from_u32(input_conv);
        for (col, &add_carry) in trace_row
            .mem_oc
            .input_read_add_carry
            .iter_mut()
            .zip(input_add.iter())
        {
            *col = F::from_u32(add_carry);
        }
        // The write reuses the converted `buffer` base cell pointer; only register the per-block
        // write add carries (and their range checks). The base conversion carry is already filled
        // above for the buffer read group.
        {
            let byte_limbs = u32_to_ptr_limbs(buffer);
            let (_conv_carry, base_cell) = byte_ptr_limbs_to_cell_ptr_limbs_value(byte_limbs);
            let buffer_write_add = compute_block_add_carries(
                &self.range_checker_chip,
                base_cell.map(|limb| limb as u16),
                KECCAK_RATE_MEM_OPS,
                cell_stride,
            );
            for (col, &add_carry) in trace_row
                .mem_oc
                .buffer_write_add_carry
                .iter_mut()
                .zip(buffer_write_add.iter())
            {
                *col = F::from_u32(add_carry);
            }
        }
        Ok(())
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
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();
    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        chip.inner
            .replay_and_fill_trace_row(postflight, step, &mem_helper, row)
    })?;
    Ok(trace)
}
