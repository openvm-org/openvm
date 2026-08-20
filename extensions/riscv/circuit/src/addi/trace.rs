use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluWImmOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_addi, AddIChip, AddICoreCols, AddIWChip};
use crate::adapters::{
    BaseAluImmU16AdapterCols, BaseAluImmU16AdapterFiller, BaseAluWImmU16AdapterCols,
    BaseAluWImmU16AdapterFiller, U16_BITS, WORD_U16_LIMBS,
};

/// Generates the RV64 ADDI trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &AddIChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(BaseAluImmOpcode::ADDI.global_opcode());
    let adapter_width = BaseAluImmU16AdapterCols::<F>::width();
    let width = adapter_width + AddICoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let immediate = postflight.instruction(step).c.as_u32();
        let imm_low11 = (immediate & 0x7ff) as u16;
        let imm_sign = ((immediate >> 11) & 1) as u16;
        let (rs1, rd) = BaseAluImmU16AdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |rs1, _| run_addi::<BLOCK_FE_WIDTH, U16_BITS>(&rs1, imm_low11, imm_sign),
        )?;

        let core_row: &mut AddICoreCols<F, BLOCK_FE_WIDTH, U16_BITS> = core_row.borrow_mut();
        core_row.is_valid = F::ONE;
        core_row.imm_sign = F::from_u16(imm_sign);
        core_row.imm_low11 = F::from_u16(imm_low11);
        chip.inner
            .range_checker_chip
            .add_count(imm_low11 as u32, 11);
        core_row.rs1 = rs1.map(F::from_u16);
        core_row.rd = rd.map(F::from_u16);
        for &value in &rd {
            chip.inner
                .range_checker_chip
                .add_count(value as u32, U16_BITS);
        }
        Ok(())
    })?;

    Ok(trace)
}

/// Generates the RV64 ADDIW trace directly from immutable preflight history.
pub fn generate_w_trace_from_postflight<F: PrimeField32>(
    chip: &AddIWChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(BaseAluWImmOpcode::ADDIW.global_opcode());
    let adapter_width = BaseAluWImmU16AdapterCols::<F>::width();
    let width = adapter_width + AddICoreCols::<F, WORD_U16_LIMBS, U16_BITS>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = BaseAluWImmU16AdapterFiller::new(chip.inner.range_checker_chip.clone());

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let immediate = postflight.instruction(step).c.as_u32();
        let imm_low11 = (immediate & 0x7ff) as u16;
        let imm_sign = ((immediate >> 11) & 1) as u16;
        let (rs1, rd) = adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |rs1, _| run_addi::<WORD_U16_LIMBS, U16_BITS>(&rs1, imm_low11, imm_sign),
        )?;

        let core_row: &mut AddICoreCols<F, WORD_U16_LIMBS, U16_BITS> = core_row.borrow_mut();
        core_row.is_valid = F::ONE;
        core_row.imm_sign = F::from_u16(imm_sign);
        core_row.imm_low11 = F::from_u16(imm_low11);
        chip.inner
            .range_checker_chip
            .add_count(imm_low11 as u32, 11);
        core_row.rs1 = rs1.map(F::from_u16);
        core_row.rd = rd.map(F::from_u16);
        for &value in &rd[..WORD_U16_LIMBS - 1] {
            chip.inner
                .range_checker_chip
                .add_count(value as u32, U16_BITS);
        }
        Ok(())
    })?;

    Ok(trace)
}
