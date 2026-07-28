use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::MulHOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    core::{fill_core_row_with_result, run_mulh},
    MulHCoreCols, Rv64MulHChip,
};
use crate::adapters::{
    Rv64MultAdapterCols, Rv64MultAdapterFiller, RV64_BYTE_BITS, RV64_REGISTER_NUM_LIMBS,
};

/// Generates the RV64 multiply-high trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64MulHChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [MulHOpcode::MULH, MulHOpcode::MULHSU, MulHOpcode::MULHU];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64MultAdapterCols::<F>::width();
    let width = adapter_width + MulHCoreCols::<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for opcode in opcodes {
        for &step in postflight.steps(opcode.global_opcode()) {
            let row = &mut trace.values[row_index * width..(row_index + 1) * width];
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let mut result = None;
            let ([b, c], _) = Rv64MultAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[b, c]| {
                    let computed = run_mulh::<RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS>(
                        opcode,
                        &b.map(u32::from),
                        &c.map(u32::from),
                    );
                    let output = computed.0.map(|limb| limb as u8);
                    result = Some(computed);
                    output
                },
            )?;
            let result = result.expect("multiply-high replay closure always runs");
            let core_row: &mut MulHCoreCols<F, RV64_REGISTER_NUM_LIMBS, RV64_BYTE_BITS> =
                core_row.borrow_mut();
            fill_core_row_with_result(
                &chip.inner.range_tuple_chip,
                &chip.inner.bitwise_lookup_chip,
                core_row,
                opcode,
                b,
                c,
                result,
            );
            row_index += 1;
        }
    }

    Ok(trace)
}
