use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    run_shift_right_arithmetic, Rv64ShiftRightArithmeticChip, ShiftRightArithmeticCoreCols,
};
use crate::adapters::{Rv64BaseAluRegU16AdapterCols, Rv64BaseAluRegU16AdapterFiller, U16_BITS};

/// Generates the RV64 arithmetic-right-shift trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64ShiftRightArithmeticChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = ShiftOpcode::SRA.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = Rv64BaseAluRegU16AdapterCols::<F>::width();
    let width =
        adapter_width + ShiftRightArithmeticCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    for (row_index, &step) in postflight.steps(opcode).iter().enumerate() {
        let row = &mut trace.values[row_index * width..(row_index + 1) * width];
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let mut shifts = (0, 0);
        let ([rs1, rs2], output) = Rv64BaseAluRegU16AdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |[rs1, rs2]| {
                let (output, limb_shift, bit_shift) =
                    run_shift_right_arithmetic::<BLOCK_FE_WIDTH, U16_BITS>(&rs1, &rs2);
                shifts = (limb_shift, bit_shift);
                output
            },
        )?;
        let core_row: &mut ShiftRightArithmeticCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
            core_row.borrow_mut();
        let (limb_shift, bit_shift) = shifts;
        let num_bits_log = (BLOCK_FE_WIDTH * U16_BITS).ilog2();
        chip.inner.range_checker_chip.add_count(
            ((rs2[0] as usize - bit_shift - limb_shift * U16_BITS) >> num_bits_log) as u32,
            U16_BITS - num_bits_log as usize,
        );

        let aux_bits = U16_BITS - bit_shift;
        for (limb, (carry_col, aux_col)) in rs1.iter().copied().zip(
            core_row
                .bit_shift_carry
                .iter_mut()
                .zip(core_row.bit_shift_aux.iter_mut()),
        ) {
            let limb = limb as u32;
            let carry = limb & ((1u32 << bit_shift) - 1);
            let aux = limb >> bit_shift;
            chip.inner.range_checker_chip.add_count(carry, bit_shift);
            chip.inner.range_checker_chip.add_count(aux, aux_bits);
            *carry_col = F::from_u32(carry);
            *aux_col = F::from_u32(aux);
        }

        core_row.limb_shift_marker[limb_shift] = F::ONE;
        core_row.bit_shift_marker[bit_shift] = F::ONE;
        let b_sign = rs1[BLOCK_FE_WIDTH - 1] >> (U16_BITS - 1);
        chip.inner.range_checker_chip.add_count(
            (rs1[BLOCK_FE_WIDTH - 1] as u32) - ((b_sign as u32) << (U16_BITS - 1)),
            U16_BITS - 1,
        );

        core_row.b_sign = F::from_u16(b_sign);
        core_row.c = rs2.map(F::from_u16);
        core_row.b = rs1.map(F::from_u16);
        core_row.a = output.map(F::from_u16);
    }

    Ok(trace)
}
