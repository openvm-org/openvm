use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::AuipcOpcode::AUIPC;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_auipc, AuipcChip, AuipcCoreCols};
use crate::adapters::{
    sext32_to_u64, RdWriteAdapterCols, RdWriteAdapterFiller, BYTE_BITS, U16_BITS,
};

/// Generates the AUIPC trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &AuipcChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(AUIPC.global_opcode());
    let adapter_width = RdWriteAdapterCols::<F>::width();
    let width = adapter_width + AuipcCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let instruction = postflight.instruction(step);
        let from_pc = postflight.pc(step);
        let immediate = instruction.c.as_canonical_u32();
        if immediate >= 1 << 24 {
            return Err(PostflightError::new(
                "AUIPC immediate exceeds its 24-bit instruction encoding",
            ));
        }
        // The circuit sign-extends negative results but cannot represent results at or above
        // 2^32 (a positive-immediate carry out of the low 32 bits is unsatisfiable).
        let result = from_pc as i64 + sext32_to_u64(immediate << BYTE_BITS) as i64;
        if result > u32::MAX as i64 {
            return Err(PostflightError::new(
                "AUIPC result exceeds implemented PC address space",
            ));
        }
        let (rd_data, _) = RdWriteAdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |pc, imm| (run_auipc(pc, imm), pc.wrapping_add(DEFAULT_PC_STEP)),
        )?;

        let core_row: &mut AuipcCoreCols<F> = core_row.borrow_mut();
        let imm_bytes = immediate.to_le_bytes();
        let imm_low_8 = imm_bytes[0];
        let imm_high_16 = (imm_bytes[1] as u32) | ((imm_bytes[2] as u32) << BYTE_BITS);
        let rd_lo = rd_data[0];
        let rd_hi = rd_data[1];
        let is_sign_extend = rd_data[2] != 0;
        let imm_sign = (imm_high_16 >> (U16_BITS - 1)) & 1;
        let imm_magnitude_check = 2u32 * imm_high_16 - imm_sign * (1 << U16_BITS);

        chip.inner
            .range_checker_chip
            .add_count(imm_low_8 as u32, BYTE_BITS);
        chip.inner
            .range_checker_chip
            .add_count(imm_high_16, U16_BITS);
        chip.inner
            .range_checker_chip
            .add_count(rd_lo as u32, U16_BITS);
        chip.inner
            .range_checker_chip
            .add_count(rd_hi as u32, U16_BITS);
        chip.inner
            .range_checker_chip
            .add_count(imm_magnitude_check, U16_BITS);

        core_row.is_valid = F::ONE;
        core_row.is_sign_extend = F::from_bool(is_sign_extend);
        core_row.imm_low_8 = F::from_u8(imm_low_8);
        core_row.imm_high_16 = F::from_u32(imm_high_16);
        core_row.rd_data = [F::from_u16(rd_lo), F::from_u16(rd_hi)];
        Ok(())
    })?;

    Ok(trace)
}
