use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    system::program::trace::instruction_operand_to_field,
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{
    program::{DEFAULT_PC_STEP, MAX_ALLOWED_PC, PC_STEP_BITS},
    LocalOpcode,
};
use openvm_riscv_transpiler::JalLuiOpcode::{self, JAL, LUI};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{get_signed_imm, run_jal_lui, JalLuiChip, JalLuiCoreCols, LUI_IMM_LOW_BITS};
use crate::adapters::{
    CondRdWriteAdapterCols, CondRdWriteAdapterFiller, PC_IDX_LOW_BITS, U16_BITS,
};

/// Generates the JAL/LUI trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &JalLuiChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [JAL, LUI];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = CondRdWriteAdapterCols::<F>::width();
    let width = adapter_width + JalLuiCoreCols::<F>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let instruction = postflight.instruction(step);
            let is_jal = local_opcode == JalLuiOpcode::JAL;
            let signed_imm = get_signed_imm(is_jal, instruction.c)
                .ok_or_else(|| PostflightError::new("JAL/LUI instruction has invalid immediate"))?;
            if is_jal {
                let from_pc = postflight.pc(step);
                if from_pc >= MAX_ALLOWED_PC {
                    return Err(PostflightError::new(
                        "JAL return address exceeds implemented PC address space",
                    ));
                }
                let target = from_pc as i64 + signed_imm as i64;
                if target < 0
                    || target > MAX_ALLOWED_PC as i64
                    || target % DEFAULT_PC_STEP as i64 != 0
                {
                    return Err(PostflightError::new(
                        "JAL target outside implemented PC address space",
                    ));
                }
            }
            let (rd_data, _) = CondRdWriteAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |from_pc| {
                    let (next_pc, output) = run_jal_lui(is_jal, from_pc, signed_imm);
                    (output, next_pc)
                },
            )?;

            let core_row: &mut JalLuiCoreCols<F> = core_row.borrow_mut();
            let rd_lo = rd_data[0];
            let rd_hi = rd_data[1];
            // JAL return addresses are zero-extended; only LUI sign-extends bit 31.
            let is_sign_extend = if is_jal {
                0
            } else {
                (rd_hi >> (U16_BITS - 1)) & 1
            };
            let imm_low_4 = if is_jal {
                0
            } else {
                (instruction.c.as_u32() & 0xf) as u8
            };

            chip.inner
                .range_checker_chip
                .add_count(rd_lo as u32, U16_BITS);
            chip.inner
                .range_checker_chip
                .add_count(rd_hi as u32, U16_BITS);
            if is_jal {
                chip.inner
                    .range_checker_chip
                    .add_count((rd_lo as u32) >> PC_STEP_BITS, PC_IDX_LOW_BITS);
            } else {
                let sign_check = 2u32 * (rd_hi as u32) - (is_sign_extend as u32) * (1 << U16_BITS);
                chip.inner
                    .range_checker_chip
                    .add_count(sign_check, U16_BITS);
                chip.inner
                    .range_checker_chip
                    .add_count(imm_low_4 as u32, LUI_IMM_LOW_BITS);
            }

            core_row.imm = instruction_operand_to_field(instruction.c);
            core_row.rd_data = [F::from_u16(rd_lo), F::from_u16(rd_hi)];
            core_row.imm_low_4 = F::from_u8(imm_low_4);
            core_row.is_jal = F::from_bool(is_jal);
            core_row.is_lui = F::from_bool(!is_jal);
            core_row.is_sign_extend = F::from_bool(is_sign_extend != 0);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
