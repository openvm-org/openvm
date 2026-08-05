use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{ShiftOpcode, ShiftWOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{ShiftWLogicalChip, ShiftWRightArithmeticChip};
use crate::{
    adapters::{BaseAluWRegU16AdapterCols, BaseAluWRegU16AdapterFiller, U16_BITS, WORD_U16_LIMBS},
    shift_logical::{run_shift_logical, ShiftLogicalCoreCols},
    shift_right_arithmetic::{run_shift_right_arithmetic, ShiftRightArithmeticCoreCols},
};

/// Generates the SLLW/SRLW trace directly from immutable preflight history.
pub fn generate_logical_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftWLogicalChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [ShiftWOpcode::SLLW, ShiftWOpcode::SRLW];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BaseAluWRegU16AdapterCols::<F>::width();
    let width = adapter_width + ShiftLogicalCoreCols::<F, WORD_U16_LIMBS, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = BaseAluWRegU16AdapterFiller::new(chip.inner.range_checker_chip.clone());

    let mut row_index = 0;
    for local_opcode in opcodes {
        let core_opcode = match local_opcode {
            ShiftWOpcode::SLLW => ShiftOpcode::SLL,
            ShiftWOpcode::SRLW => ShiftOpcode::SRL,
            ShiftWOpcode::SRAW => unreachable!(),
        };
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let mut shifts = (0, 0);
            let ([rs1, rs2], output) = adapter.replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |[rs1, rs2]| {
                    let (output, limb_shift, bit_shift) =
                        run_shift_logical::<WORD_U16_LIMBS, U16_BITS>(core_opcode, &rs1, &rs2);
                    shifts = (limb_shift, bit_shift);
                    output
                },
            )?;

            let (limb_shift, bit_shift) = shifts;
            let num_bits_log = (WORD_U16_LIMBS * U16_BITS).ilog2();
            chip.inner.range_checker_chip.add_count(
                ((rs2[0] as usize - bit_shift - limb_shift * U16_BITS) >> num_bits_log) as u32,
                U16_BITS - num_bits_log as usize,
            );

            let is_sll = local_opcode == ShiftWOpcode::SLLW;
            let aux_bits = U16_BITS - bit_shift;
            let mut bit_shift_carry = [F::ZERO; WORD_U16_LIMBS];
            let mut bit_shift_aux = [F::ZERO; WORD_U16_LIMBS];
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

            let core_row: &mut ShiftLogicalCoreCols<F, WORD_U16_LIMBS, U16_BITS> =
                core_row.borrow_mut();
            core_row.limb_shift_marker[limb_shift] = F::ONE;
            core_row.bit_shift_marker[bit_shift] = F::ONE;
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
            core_row.c = rs2.map(F::from_u16);
            core_row.b = rs1.map(F::from_u16);
            core_row.a = output.map(F::from_u16);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}

/// Generates the SRAW trace directly from immutable preflight history.
pub fn generate_right_arithmetic_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftWRightArithmeticChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = ShiftWOpcode::SRAW.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = BaseAluWRegU16AdapterCols::<F>::width();
    let width =
        adapter_width + ShiftRightArithmeticCoreCols::<F, WORD_U16_LIMBS, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = BaseAluWRegU16AdapterFiller::new(chip.inner.range_checker_chip.clone());

    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let mut shifts = (0, 0);
        let ([rs1, rs2], output) = adapter.replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |[rs1, rs2]| {
                let (output, limb_shift, bit_shift) =
                    run_shift_right_arithmetic::<WORD_U16_LIMBS, U16_BITS>(&rs1, &rs2);
                shifts = (limb_shift, bit_shift);
                output
            },
        )?;
        let (limb_shift, bit_shift) = shifts;
        let num_bits_log = (WORD_U16_LIMBS * U16_BITS).ilog2();
        chip.inner.range_checker_chip.add_count(
            ((rs2[0] as usize - bit_shift - limb_shift * U16_BITS) >> num_bits_log) as u32,
            U16_BITS - num_bits_log as usize,
        );

        let core_row: &mut ShiftRightArithmeticCoreCols<F, WORD_U16_LIMBS, U16_BITS> =
            core_row.borrow_mut();
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
        let b_sign = rs1[WORD_U16_LIMBS - 1] >> (U16_BITS - 1);
        chip.inner.range_checker_chip.add_count(
            (rs1[WORD_U16_LIMBS - 1] as u32) - ((b_sign as u32) << (U16_BITS - 1)),
            U16_BITS - 1,
        );
        core_row.b_sign = F::from_u16(b_sign);
        core_row.c = rs2.map(F::from_u16);
        core_row.b = rs1.map(F::from_u16);
        core_row.a = output.map(F::from_u16);
        Ok(())
    })?;

    Ok(trace)
}
