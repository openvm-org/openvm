use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::LessThanImmOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{imm_to_u16_limbs, LessThanImmChip, LessThanImmCoreCols};
use crate::{
    adapters::{BaseAluImmU16AdapterCols, BaseAluImmU16AdapterFiller, U16_BITS},
    less_than::run_less_than,
};

/// Generates the RV64 immediate less-than trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &LessThanImmChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [LessThanImmOpcode::SLTI, LessThanImmOpcode::SLTIU];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BaseAluImmU16AdapterCols::<F>::width();
    let width = adapter_width + LessThanImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let instruction = postflight.instruction(step);
            let immediate = instruction.c.as_canonical_u32();
            let imm_low11 = (immediate & 0x7ff) as u16;
            let imm_sign = ((immediate >> 11) & 1) as u8;
            let c = imm_to_u16_limbs::<BLOCK_FE_WIDTH>(imm_low11, imm_sign);
            let is_slt = local_opcode == LessThanImmOpcode::SLTI;
            let mut comparison = (false, 0, false, false);
            let (b, _) = BaseAluImmU16AdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |b, _| {
                    comparison = run_less_than::<BLOCK_FE_WIDTH, U16_BITS>(is_slt, &b, &c);
                    let mut output = [0; BLOCK_FE_WIDTH];
                    output[0] = comparison.0 as u16;
                    output
                },
            )?;
            let core_row: &mut LessThanImmCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
                core_row.borrow_mut();
            let (cmp_result, diff_idx, b_sign, c_sign) = comparison;

            let (b_msb_f, b_msb_range) = if b_sign {
                (
                    -F::from_u16(b[BLOCK_FE_WIDTH - 1].wrapping_neg()),
                    b[BLOCK_FE_WIDTH - 1] as u32 - (1u32 << (U16_BITS - 1)),
                )
            } else {
                (
                    F::from_u16(b[BLOCK_FE_WIDTH - 1]),
                    b[BLOCK_FE_WIDTH - 1] as u32 + ((is_slt as u32) << (U16_BITS - 1)),
                )
            };
            let c_msb_f = if c_sign && is_slt {
                -F::ONE
            } else {
                F::from_u16(c[BLOCK_FE_WIDTH - 1])
            };

            core_row.diff_val = if diff_idx == BLOCK_FE_WIDTH {
                F::ZERO
            } else if diff_idx == BLOCK_FE_WIDTH - 1 {
                if cmp_result {
                    c_msb_f - b_msb_f
                } else {
                    b_msb_f - c_msb_f
                }
            } else if cmp_result {
                F::from_u16((c[diff_idx] as u32 - b[diff_idx] as u32) as u16)
            } else {
                F::from_u16((b[diff_idx] as u32 - c[diff_idx] as u32) as u16)
            };

            chip.inner
                .range_checker_chip
                .add_count(imm_low11 as u32, 11);
            chip.inner
                .range_checker_chip
                .add_count(b_msb_range, U16_BITS);
            core_row.diff_marker = [F::ZERO; BLOCK_FE_WIDTH];
            if diff_idx != BLOCK_FE_WIDTH {
                chip.inner
                    .range_checker_chip
                    .add_count(core_row.diff_val.as_canonical_u32() - 1, U16_BITS);
                core_row.diff_marker[diff_idx] = F::ONE;
            }

            core_row.b_msb_f = b_msb_f;
            core_row.opcode_sltu_flag = F::from_bool(!is_slt);
            core_row.opcode_slt_flag = F::from_bool(is_slt);
            core_row.cmp_result = F::from_bool(cmp_result);
            core_row.imm_sign = F::from_u8(imm_sign);
            core_row.imm_low11 = F::from_u16(imm_low11);
            core_row.b = b.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
