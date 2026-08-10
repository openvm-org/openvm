use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::DEFAULT_PC_STEP, riscv::REGISTER_AS, LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{RevealChip, RevealCols};
use crate::adapters::{
    address_add_imm, byte_ptr_to_u16_ptr_value, checked_register_pointer, ptr_to_field_u16_limbs,
    ptr_to_u16_limbs, sign_extend_imm16, u16_block_to_bytes, PTR_U16_LIMBS, U16_BITS,
};

pub(crate) fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &RevealChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(RevealOpcode::REVEAL.global_opcode());
    let width = RevealCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let instruction = postflight.instruction(step);
        if instruction.opcode != RevealOpcode::REVEAL.global_opcode()
            || instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != PUBLIC_VALUES_AS
            || !instruction.f.is_one()
        {
            return Err(PostflightError::new(
                "REVEAL instruction has invalid fixed operands",
            ));
        }
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "REVEAL instruction has a non-boolean immediate sign",
                ));
            }
        };
        let imm = instruction.c.as_canonical_u32();
        if imm > u16::MAX as u32 {
            return Err(PostflightError::new(
                "REVEAL immediate exceeds the u16 execution-bus operand",
            ));
        }

        let base_ptr = u32::from(checked_register_pointer(instruction.b.as_canonical_u32())?);
        let src_ptr = u32::from(checked_register_pointer(instruction.a.as_canonical_u32())?);
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let mut replay = postflight.replay(step);
        let base = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(base_ptr))?;
        if base.value[PTR_U16_LIMBS..].iter().any(|&limb| limb != 0) {
            return Err(PostflightError::new(
                "REVEAL base register is not a low-32-bit pointer",
            ));
        }
        let base_value = u32::from(base.value[0]) | (u32::from(base.value[1]) << U16_BITS);
        let dst_ptr = address_add_imm(base_value, sign_extend_imm16(imm, u32::from(imm_sign)));
        let pointer_limit = 1u64
            .checked_shl(chip.inner.pointer_max_bits as u32)
            .unwrap_or(u64::MAX);
        if dst_ptr > u64::from(u32::MAX)
            || !dst_ptr.is_multiple_of(MEMORY_BLOCK_BYTES as u64)
            || dst_ptr + MEMORY_BLOCK_BYTES as u64 > pointer_limit
        {
            return Err(PostflightError::new(
                "REVEAL destination is not an aligned in-bounds pointer",
            ));
        }
        let dst_ptr = dst_ptr as u32;

        let src = replay.read_u16(REGISTER_AS, byte_ptr_to_u16_ptr_value(src_ptr))?;
        let src_bytes = u16_block_to_bytes(src.value);
        let writes = [
            replay.write_u8(
                PUBLIC_VALUES_AS,
                dst_ptr,
                src_bytes[..BLOCK_FE_WIDTH].try_into().unwrap(),
            )?,
            replay.write_u8(
                PUBLIC_VALUES_AS,
                dst_ptr + BLOCK_FE_WIDTH as u32,
                src_bytes[BLOCK_FE_WIDTH..].try_into().unwrap(),
            )?,
        ];
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        let dst_ptr_limbs = ptr_to_u16_limbs(dst_ptr).map(u32::from);
        chip.inner
            .range_checker_chip
            .add_count(dst_ptr_limbs[0] >> 3, U16_BITS - 3);
        chip.inner
            .range_checker_chip
            .add_count(dst_ptr_limbs[1], chip.inner.pointer_max_bits - U16_BITS);
        for bytes in src_bytes.chunks_exact(2) {
            chip.inner
                .bitwise_lookup_chip
                .request_range(u32::from(bytes[0]), u32::from(bytes[1]));
        }

        let cols: &mut RevealCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.from_state.pc = F::from_u32(from_pc);
        cols.from_state.timestamp = F::from_u32(from_timestamp);
        cols.base_ptr = F::from_u32(base_ptr);
        cols.base_ptr_limbs = ptr_to_field_u16_limbs(base_value);
        mem_helper.fill(
            base.previous_timestamp,
            base.timestamp,
            cols.base_aux.as_mut(),
        );
        cols.src_ptr = F::from_u32(src_ptr);
        cols.src_bytes = src_bytes.map(F::from_u8);
        mem_helper.fill(src.previous_timestamp, src.timestamp, cols.src_aux.as_mut());
        cols.imm = F::from_u32(imm);
        cols.imm_sign = F::from_bool(imm_sign);
        cols.dst_ptr_low_limb = F::from_u32(dst_ptr_limbs[0]);
        for (aux, write) in cols.write_aux.iter_mut().zip(writes) {
            aux.set_prev_data(write.previous_value.map(F::from_u8));
            mem_helper.fill(write.previous_timestamp, write.timestamp, aux.as_mut());
        }
        Ok(())
    })?;
    Ok(trace)
}
