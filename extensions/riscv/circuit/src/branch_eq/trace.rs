use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::padded_trace_height,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BranchEqualOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{fast_run_eq, run_eq, BranchEqualCoreCols, Rv64BranchEqualChip};
use crate::adapters::{Rv64BranchAdapterCols, Rv64BranchAdapterFiller};

/// Generates the RV64 equality-branch trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64BranchEqualChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [BranchEqualOpcode::BEQ, BranchEqualOpcode::BNE];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64BranchAdapterCols::<F>::width();
    let width = adapter_width + BranchEqualCoreCols::<F, BLOCK_FE_WIDTH>::width();
    let height = padded_trace_height(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let (inputs, _) = Rv64BranchAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |from_pc, [rs1, rs2], immediate| {
                    if fast_run_eq(local_opcode, &rs1, &rs2) {
                        (F::from_u32(from_pc) + F::from_u32(immediate)).as_canonical_u32()
                    } else {
                        from_pc.wrapping_add(chip.inner.pc_step)
                    }
                },
            )?;
            let [a, b] = inputs;
            let core_row: &mut BranchEqualCoreCols<F, BLOCK_FE_WIDTH> = core_row.borrow_mut();
            let instruction = postflight.instruction(step);
            let is_beq = local_opcode == BranchEqualOpcode::BEQ;
            let (cmp_result, diff_idx, diff_inv_val) = run_eq::<F, BLOCK_FE_WIDTH>(is_beq, &a, &b);

            core_row.diff_inv_marker = [F::ZERO; BLOCK_FE_WIDTH];
            core_row.diff_inv_marker[diff_idx] = diff_inv_val;
            core_row.opcode_bne_flag = F::from_bool(!is_beq);
            core_row.opcode_beq_flag = F::from_bool(is_beq);
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
