use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::{pc_to_idx, DEFAULT_PC_STEP},
    LocalOpcode,
};
use openvm_riscv_transpiler::AuipcOpcode::AUIPC;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_auipc, AuipcChip, AuipcCoreCols};
use crate::adapters::{
    RdWriteAdapterCols, RdWriteAdapterFiller, BYTE_BITS, PC_IDX_LOW_BITS, U16_BITS,
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
        let immediate = instruction.c.as_u32();
        if immediate >= 1 << 24 {
            return Err(PostflightError::new(
                "AUIPC immediate exceeds its 24-bit instruction encoding",
            ));
        }
        let (rd_data, _) = RdWriteAdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |pc| (run_auipc(pc, immediate), pc.wrapping_add(DEFAULT_PC_STEP)),
        )?;

        let core_row: &mut AuipcCoreCols<F> = core_row.borrow_mut();
        let imm_bytes = immediate.to_le_bytes();
        let imm_low_8 = imm_bytes[0];
        let imm_high_16 = (imm_bytes[1] as u32) | ((imm_bytes[2] as u32) << BYTE_BITS);
        let pc_idx = pc_to_idx(from_pc);
        let pc_idx_low = pc_idx & ((1 << PC_IDX_LOW_BITS) - 1);
        let pc_high = pc_idx >> PC_IDX_LOW_BITS;
        let rd_lo = rd_data[0];
        let rd_hi = rd_data[1];
        let imm_sign = (imm_high_16 >> (U16_BITS - 1)) & 1;
        let imm_magnitude_check = 2u32 * imm_high_16 - imm_sign * (1 << U16_BITS);

        chip.inner
            .range_checker_chip
            .add_count(pc_idx_low, PC_IDX_LOW_BITS);
        chip.inner.range_checker_chip.add_count(pc_high, U16_BITS);
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
        core_row.imm_sign = F::from_bool(imm_sign != 0);
        core_row.imm_low_8 = F::from_u8(imm_low_8);
        core_row.imm_high_16 = F::from_u32(imm_high_16);
        core_row.pc_high = F::from_u32(pc_high);
        core_row.rd_data = [F::from_u16(rd_lo), F::from_u16(rd_hi)];
        Ok(())
    })?;

    Ok(trace)
}
