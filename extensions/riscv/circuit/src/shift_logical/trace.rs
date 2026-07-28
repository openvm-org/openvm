use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::ShiftOpcode;
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{run_shift_logical, Rv64ShiftLogicalChip, ShiftLogicalCoreCols};
use crate::adapters::{Rv64BaseAluRegU16AdapterCols, Rv64BaseAluRegU16AdapterFiller, U16_BITS};

/// Generates the RV64 logical-shift trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &Rv64ShiftLogicalChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [ShiftOpcode::SLL, ShiftOpcode::SRL];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = Rv64BaseAluRegU16AdapterCols::<F>::width();
    let width = adapter_width + ShiftLogicalCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        for &step in postflight.steps(local_opcode.global_opcode()) {
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
                        run_shift_logical::<BLOCK_FE_WIDTH, U16_BITS>(local_opcode, &rs1, &rs2);
                    shifts = (limb_shift, bit_shift);
                    output
                },
            )?;

            let (limb_shift, bit_shift) = shifts;
            let num_bits_log = (BLOCK_FE_WIDTH * U16_BITS).ilog2();
            chip.inner.range_checker_chip.add_count(
                ((rs2[0] as usize - bit_shift - limb_shift * U16_BITS) >> num_bits_log) as u32,
                U16_BITS - num_bits_log as usize,
            );

            let is_sll = local_opcode == ShiftOpcode::SLL;
            let aux_bits = U16_BITS - bit_shift;
            let mut bit_shift_carry = [F::ZERO; BLOCK_FE_WIDTH];
            let mut bit_shift_aux = [F::ZERO; BLOCK_FE_WIDTH];
            for (limb, (carry_cell, aux_cell)) in rs1
                .iter()
                .zip(bit_shift_carry.iter_mut().zip(&mut bit_shift_aux))
            {
                let limb = *limb as u32;
                let (carry, aux) = if is_sll {
                    (limb >> aux_bits, limb & ((1u32 << aux_bits) - 1))
                } else {
                    (limb & ((1u32 << bit_shift) - 1), limb >> bit_shift)
                };
                chip.inner.range_checker_chip.add_count(carry, bit_shift);
                chip.inner.range_checker_chip.add_count(aux, aux_bits);
                *carry_cell = F::from_u32(carry);
                *aux_cell = F::from_u32(aux);
            }

            let mut limb_shift_marker = [F::ZERO; BLOCK_FE_WIDTH];
            limb_shift_marker[limb_shift] = F::ONE;
            let mut bit_shift_marker = [F::ZERO; U16_BITS];
            bit_shift_marker[bit_shift] = F::ONE;

            let core_row: &mut ShiftLogicalCoreCols<F, BLOCK_FE_WIDTH, U16_BITS> =
                core_row.borrow_mut();
            core_row.carry_multiplier_left = if is_sll {
                F::from_u32(1 << aux_bits)
            } else {
                F::ZERO
            };
            core_row.bit_multiplier_left = if is_sll {
                F::from_u32(1 << bit_shift)
            } else {
                F::ZERO
            };
            core_row.opcode_sll_flag = F::from_bool(is_sll);
            core_row.bit_shift_aux = bit_shift_aux;
            core_row.bit_shift_carry = bit_shift_carry;
            core_row.limb_shift_marker = limb_shift_marker;
            core_row.bit_shift_marker = bit_shift_marker;
            core_row.c = rs2.map(F::from_u16);
            core_row.b = rs1.map(F::from_u16);
            core_row.a = output.map(F::from_u16);
            row_index += 1;
        }
    }

    Ok(trace)
}
