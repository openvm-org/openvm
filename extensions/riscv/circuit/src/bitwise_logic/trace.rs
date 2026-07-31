use std::{borrow::BorrowMut, iter::zip};

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_bitwise_logic, BitwiseLogicCoreCols, Rv64BitwiseLogicChip, RV64_BYTE_BITS};
use crate::adapters::{Rv64BaseAluRegAdapterCols, Rv64BaseAluRegAdapterFiller};

/// Generates the RV64 bitwise trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64BitwiseLogicChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [BaseAluOpcode::XOR, BaseAluOpcode::OR, BaseAluOpcode::AND];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64BaseAluRegAdapterCols::<F>::width();
    let width =
        adapter_width + BitwiseLogicCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let ([rs1, rs2], output) = Rv64BaseAluRegAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[rs1, rs2]| {
                    run_bitwise_logic::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(
                        local_opcode,
                        &rs1,
                        &rs2,
                    )
                },
            )?;
            let core_row: &mut BitwiseLogicCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
                core_row.borrow_mut();
            core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluOpcode::AND);
            core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluOpcode::OR);
            core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluOpcode::XOR);
            for (&rs1, &rs2) in zip(&rs1, &rs2) {
                chip.inner
                    .bitwise_lookup_chip
                    .request_xor(rs1 as u32, rs2 as u32);
            }
            core_row.c = rs2.map(F::from_u8);
            core_row.b = rs1.map(F::from_u8);
            core_row.a = output.map(F::from_u8);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
