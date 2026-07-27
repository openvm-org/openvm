use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64JalLuiOpcode::{self, JAL, LUI};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    get_signed_imm, run_jal_lui, Rv64JalLuiChip, Rv64JalLuiCoreCols, LUI_IMM_LOW_BITS,
    PC_HIGH_U16_SHIFT,
};
use crate::adapters::{Rv64CondRdWriteAdapterCols, Rv64CondRdWriteAdapterFiller, U16_BITS};

/// Generates the JAL/LUI trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64JalLuiChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [JAL, LUI];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64CondRdWriteAdapterCols::<F>::width();
    let width = adapter_width + Rv64JalLuiCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        for &step in postflight.steps(local_opcode.global_opcode()) {
            let row = &mut trace.values[row_index * width..(row_index + 1) * width];
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let instruction = postflight.instruction(step);
            let is_jal = local_opcode == Rv64JalLuiOpcode::JAL;
            let signed_imm = get_signed_imm(is_jal, instruction.c);
            let (rd_data, _) = Rv64CondRdWriteAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |from_pc, _| {
                    let (next_pc, output) = run_jal_lui(is_jal, from_pc, signed_imm);
                    (output, next_pc)
                },
            )?;

            let core_row: &mut Rv64JalLuiCoreCols<F> = core_row.borrow_mut();
            let rd_lo = rd_data[0];
            let rd_hi = rd_data[1];
            let is_sign_extend = (rd_hi >> (U16_BITS - 1)) & 1;
            let sign_check = 2u32 * (rd_hi as u32) - (is_sign_extend as u32) * (1 << U16_BITS);
            let imm_low_4 = if is_jal {
                0
            } else {
                (instruction.c.as_canonical_u32() & 0xf) as u8
            };

            chip.inner
                .range_checker_chip
                .add_count(rd_lo as u32, U16_BITS);
            chip.inner
                .range_checker_chip
                .add_count(rd_hi as u32, U16_BITS);
            chip.inner
                .range_checker_chip
                .add_count(sign_check, U16_BITS);
            if is_jal {
                chip.inner
                    .range_checker_chip
                    .add_count((rd_hi as u32) << PC_HIGH_U16_SHIFT, U16_BITS);
            } else {
                chip.inner
                    .range_checker_chip
                    .add_count(imm_low_4 as u32, LUI_IMM_LOW_BITS);
            }

            core_row.imm = instruction.c;
            core_row.rd_data = [F::from_u16(rd_lo), F::from_u16(rd_hi)];
            core_row.imm_low_4 = F::from_u8(imm_low_4);
            core_row.is_jal = F::from_bool(is_jal);
            core_row.is_lui = F::from_bool(!is_jal);
            core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
            row_index += 1;
        }
    }

    Ok(trace)
}
