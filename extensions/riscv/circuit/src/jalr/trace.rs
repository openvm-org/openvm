use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::JalrOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{try_run_jalr, JalrChip, JalrCoreCols};
use crate::adapters::{u16_block_to_u64, JalrAdapterCols, JalrAdapterFiller};

/// Generates the RV64 JALR trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &JalrChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let steps = postflight.steps(JalrOpcode::JALR.global_opcode());
    let adapter_width = JalrAdapterCols::<F>::width();
    let width = adapter_width + JalrCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(steps.len());
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, steps, |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let (rs1, raw_target_pc, rd_data) = JalrAdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |from_pc, rs1, immediate, imm_sign| {
                try_run_jalr(from_pc, u16_block_to_u64(rs1), immediate, imm_sign).ok_or_else(|| {
                    PostflightError::new(
                        "JALR target is outside implemented PC address space or misaligned",
                    )
                })
            },
        )?;
        let instruction = postflight.instruction(step);
        chip.inner.fill_core_row(
            core_row.borrow_mut(),
            u16_block_to_u64(rs1),
            instruction.c.as_u32() as u16,
            instruction.g.is_one(),
            raw_target_pc,
            rd_data,
        );
        Ok(())
    })?;

    Ok(trace)
}
