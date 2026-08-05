use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::BaseAluOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_add_sub, AddSubChip, AddSubCoreCols};
use crate::adapters::{BaseAluRegU16AdapterCols, BaseAluRegU16AdapterFiller, U16_BITS};

/// Generates the RV64 ADD/SUB trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &AddSubChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let add = BaseAluOpcode::ADD.global_opcode();
    let sub = BaseAluOpcode::SUB.global_opcode();
    let rows_used = postflight.steps(add).len() + postflight.steps(sub).len();
    let adapter_width = BaseAluRegU16AdapterCols::<F>::width();
    let width = adapter_width + AddSubCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let mem_helper = chip.mem_helper.as_borrowed();

    let mut row_index = 0;
    for (opcode, local_opcode) in [(add, BaseAluOpcode::ADD), (sub, BaseAluOpcode::SUB)] {
        let steps = postflight.steps(opcode);
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let ([rs1, rs2], output) = BaseAluRegU16AdapterFiller::replay(
                postflight,
                step,
                &mem_helper,
                adapter_row.borrow_mut(),
                |[rs1, rs2]| run_add_sub::<BLOCK_FE_WIDTH, U16_BITS>(local_opcode, &rs1, &rs2),
            )?;
            let core_row: &mut AddSubCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> = core_row.borrow_mut();
            core_row.opcode_sub_flag = F::from_bool(local_opcode == BaseAluOpcode::SUB);
            core_row.opcode_add_flag = F::from_bool(local_opcode == BaseAluOpcode::ADD);
            for &value in &output {
                chip.inner
                    .range_checker_chip
                    .add_count(value as u32, U16_BITS);
            }
            core_row.c = rs2.map(F::from_u16);
            core_row.b = rs1.map(F::from_u16);
            core_row.a = output.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
