use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::BranchLessThanOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_cmp, BranchLessThanChip, BranchLessThanCoreCols};
use crate::adapters::{BranchAdapterCols, BranchAdapterFiller, U16_BITS};

/// Generates the RV64 less-than-branch trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &BranchLessThanChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [
        BranchLessThanOpcode::BLT,
        BranchLessThanOpcode::BLTU,
        BranchLessThanOpcode::BGE,
        BranchLessThanOpcode::BGEU,
    ];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BranchAdapterCols::<F>::width();
    let width = adapter_width + BranchLessThanCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let local_opcode_u8 = local_opcode as u8;
            let mut comparison = (false, 0, false, false);
            let (inputs, _) = BranchAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |from_pc, [rs1, rs2], immediate| {
                    comparison = run_cmp::<BLOCK_FE_WIDTH, U16_BITS>(local_opcode_u8, &rs1, &rs2);
                    if comparison.0 {
                        (F::from_u32(from_pc) + F::from_u32(immediate)).as_canonical_u32()
                    } else {
                        from_pc.wrapping_add(DEFAULT_PC_STEP)
                    }
                },
            )?;
            let [a, b] = inputs;
            let core_row: &mut BranchLessThanCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
                core_row.borrow_mut();
            let instruction = postflight.instruction(step);
            let signed = matches!(
                local_opcode,
                BranchLessThanOpcode::BLT | BranchLessThanOpcode::BGE
            );
            let ge_op = matches!(
                local_opcode,
                BranchLessThanOpcode::BGE | BranchLessThanOpcode::BGEU
            );
            let (cmp_result, diff_idx, a_sign, b_sign) = comparison;
            let cmp_lt = cmp_result ^ ge_op;

            let (a_msb_f, a_msb_range) = if a_sign {
                (
                    -F::from_u16(a[BLOCK_FE_WIDTH - 1].wrapping_neg()),
                    a[BLOCK_FE_WIDTH - 1] as u32 - (1 << (U16_BITS - 1)),
                )
            } else {
                (
                    F::from_u16(a[BLOCK_FE_WIDTH - 1]),
                    a[BLOCK_FE_WIDTH - 1] as u32 + ((signed as u32) << (U16_BITS - 1)),
                )
            };
            let (b_msb_f, b_msb_range) = if b_sign {
                (
                    -F::from_u16(b[BLOCK_FE_WIDTH - 1].wrapping_neg()),
                    b[BLOCK_FE_WIDTH - 1] as u32 - (1 << (U16_BITS - 1)),
                )
            } else {
                (
                    F::from_u16(b[BLOCK_FE_WIDTH - 1]),
                    b[BLOCK_FE_WIDTH - 1] as u32 + ((signed as u32) << (U16_BITS - 1)),
                )
            };

            core_row.diff_val = if diff_idx == BLOCK_FE_WIDTH {
                F::ZERO
            } else if diff_idx == BLOCK_FE_WIDTH - 1 {
                if cmp_lt {
                    b_msb_f - a_msb_f
                } else {
                    a_msb_f - b_msb_f
                }
            } else if cmp_lt {
                F::from_u16((b[diff_idx] as u32 - a[diff_idx] as u32) as u16)
            } else {
                F::from_u16((a[diff_idx] as u32 - b[diff_idx] as u32) as u16)
            };

            chip.inner
                .range_checker_chip
                .add_count(a_msb_range, U16_BITS);
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

            core_row.cmp_lt = F::from_bool(cmp_lt);
            core_row.b_msb_f = b_msb_f;
            core_row.a_msb_f = a_msb_f;
            core_row.opcode_bgeu_flag = F::from_bool(local_opcode == BranchLessThanOpcode::BGEU);
            core_row.opcode_bge_flag = F::from_bool(local_opcode == BranchLessThanOpcode::BGE);
            core_row.opcode_bltu_flag = F::from_bool(local_opcode == BranchLessThanOpcode::BLTU);
            core_row.opcode_blt_flag = F::from_bool(local_opcode == BranchLessThanOpcode::BLT);
            core_row.imm = instruction.c;
            core_row.cmp_result = F::from_bool(cmp_result);
            core_row.b = b.map(F::from_u16);
            core_row.a = a.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
