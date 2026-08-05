use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{ShiftImmOpcode, ShiftOpcode, ShiftWImmOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{ShiftLogicalImmChip, ShiftLogicalImmCoreCols, ShiftWLogicalImmChip};
use crate::{
    adapters::{
        BaseAluImmU16AdapterCols, BaseAluImmU16AdapterFiller, BaseAluWImmU16AdapterCols,
        BaseAluWImmU16AdapterFiller, U16_BITS, WORD_U16_LIMBS,
    },
    shift_logical::run_shift_logical,
};

pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftLogicalImmChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [ShiftImmOpcode::SLLI, ShiftImmOpcode::SRLI];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BaseAluImmU16AdapterCols::<F>::width();
    let width = adapter_width + ShiftLogicalImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let instruction = postflight.instruction(step);
            let shamt = instruction.c.as_u32() as usize;
            if shamt >= BLOCK_FE_WIDTH * U16_BITS {
                return Err(PostflightError::new(
                    "logical shift immediate is out of range",
                ));
            }

            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let is_sll = local_opcode == ShiftImmOpcode::SLLI;
            let reg_opcode = if is_sll {
                ShiftOpcode::SLL
            } else {
                ShiftOpcode::SRL
            };
            let mut shifts = (0, 0);
            let (input, output) = BaseAluImmU16AdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |input, immediate| {
                    let mut shamt_limbs = [0u16; BLOCK_FE_WIDTH];
                    shamt_limbs[0] = immediate as u16;
                    let (output, limb_shift, bit_shift) = run_shift_logical::<
                        BLOCK_FE_WIDTH,
                        U16_BITS,
                    >(
                        reg_opcode, &input, &shamt_limbs
                    );
                    shifts = (limb_shift, bit_shift);
                    output
                },
            )?;

            fill_core(
                &chip.inner.range_checker_chip,
                core_row.borrow_mut(),
                is_sll,
                input,
                output,
                shifts,
            );
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}

pub fn generate_word_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftWLogicalImmChip<F>,
    postflight: &Postflight<'_>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [ShiftWImmOpcode::SLLIW, ShiftWImmOpcode::SRLIW];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BaseAluWImmU16AdapterCols::<F>::width();
    let width = adapter_width + ShiftLogicalImmCoreCols::<F, WORD_U16_LIMBS, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = BaseAluWImmU16AdapterFiller::new(chip.inner.range_checker_chip.clone());

    let mut row_index = 0;
    for local_opcode in opcodes {
        let is_sll = local_opcode == ShiftWImmOpcode::SLLIW;
        let reg_opcode = if is_sll {
            ShiftOpcode::SLL
        } else {
            ShiftOpcode::SRL
        };
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let instruction = postflight.instruction(step);
            let shamt = instruction.c.as_u32() as usize;
            if shamt >= WORD_U16_LIMBS * U16_BITS {
                return Err(PostflightError::new(
                    "word logical shift immediate is out of range",
                ));
            }

            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let mut shifts = (0, 0);
            let (input, output) = adapter.replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |input, immediate| {
                    let mut shamt_limbs = [0u16; WORD_U16_LIMBS];
                    shamt_limbs[0] = immediate as u16;
                    let (output, limb_shift, bit_shift) = run_shift_logical::<
                        WORD_U16_LIMBS,
                        U16_BITS,
                    >(
                        reg_opcode, &input, &shamt_limbs
                    );
                    shifts = (limb_shift, bit_shift);
                    output
                },
            )?;

            fill_core(
                &chip.inner.range_checker_chip,
                core_row.borrow_mut(),
                is_sll,
                input,
                output,
                shifts,
            );
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}

fn fill_core<F: PrimeField32, const NUM_LIMBS: usize>(
    range_checker: &SharedVariableRangeCheckerChip,
    core_row: &mut ShiftLogicalImmCoreCols<F, NUM_LIMBS, U16_BITS>,
    is_sll: bool,
    input: [u16; NUM_LIMBS],
    output: [u16; NUM_LIMBS],
    (limb_shift, bit_shift): (usize, usize),
) {
    let aux_bits = U16_BITS - bit_shift;
    for (limb, (carry_cell, aux_cell)) in input.iter().zip(
        core_row
            .bit_shift_carry
            .iter_mut()
            .zip(&mut core_row.bit_shift_aux),
    ) {
        let limb = *limb as u32;
        let (carry, aux) = if is_sll {
            (limb >> aux_bits, limb & ((1u32 << aux_bits) - 1))
        } else {
            (limb & ((1u32 << bit_shift) - 1), limb >> bit_shift)
        };
        range_checker.add_count(carry, bit_shift);
        range_checker.add_count(aux, aux_bits);
        *carry_cell = F::from_u32(carry);
        *aux_cell = F::from_u32(aux);
    }

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
    core_row.b = input.map(F::from_u16);
    core_row.a = output.map(F::from_u16);
}
