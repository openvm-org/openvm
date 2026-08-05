use std::borrow::BorrowMut;

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError, BLOCK_FE_WIDTH},
    utils::next_power_of_two_or_zero,
};
use openvm_circuit_primitives::var_range::SharedVariableRangeCheckerChip;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::{ShiftImmOpcode, ShiftWImmOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{
    ShiftRightArithmeticImmChip, ShiftRightArithmeticImmCoreCols, ShiftWRightArithmeticImmChip,
};
use crate::{
    adapters::{
        BaseAluImmU16AdapterCols, BaseAluImmU16AdapterFiller, BaseAluWImmU16AdapterCols,
        BaseAluWImmU16AdapterFiller, U16_BITS, WORD_U16_LIMBS,
    },
    shift_right_arithmetic::run_shift_right_arithmetic,
};

pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftRightArithmeticImmChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = ShiftImmOpcode::SRAI.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = BaseAluImmU16AdapterCols::<F>::width();
    let width =
        adapter_width + ShiftRightArithmeticImmCoreCols::<F, BLOCK_FE_WIDTH, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let instruction = postflight.instruction(step);
        let shamt = instruction.c.as_canonical_u32() as usize;
        if shamt >= BLOCK_FE_WIDTH * U16_BITS {
            return Err(PostflightError::new(
                "arithmetic shift immediate is out of range",
            ));
        }

        let (adapter_row, core_row) = row.split_at_mut(adapter_width);
        let mut shifts = (0, 0);
        let (input, output) = BaseAluImmU16AdapterFiller::replay(
            postflight,
            step,
            &chip.mem_helper.as_borrowed(),
            adapter_row.borrow_mut(),
            |input, immediate| {
                let mut shamt_limbs = [0u16; BLOCK_FE_WIDTH];
                shamt_limbs[0] = immediate as u16;
                let (output, limb_shift, bit_shift) =
                    run_shift_right_arithmetic::<BLOCK_FE_WIDTH, U16_BITS>(&input, &shamt_limbs);
                shifts = (limb_shift, bit_shift);
                output
            },
        )?;

        fill_core(
            &chip.inner.range_checker_chip,
            core_row.borrow_mut(),
            input,
            output,
            shifts,
        );
        Ok(())
    })?;

    Ok(trace)
}

pub fn generate_word_trace_from_postflight<F: PrimeField32>(
    chip: &ShiftWRightArithmeticImmChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcode = ShiftWImmOpcode::SRAIW.global_opcode();
    let rows_used = postflight.steps(opcode).len();
    let adapter_width = BaseAluWImmU16AdapterCols::<F>::width();
    let width =
        adapter_width + ShiftRightArithmeticImmCoreCols::<F, WORD_U16_LIMBS, U16_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);
    let adapter = BaseAluWImmU16AdapterFiller::new(chip.inner.range_checker_chip.clone());

    fill_trace_rows(&mut trace, 0, postflight.steps(opcode), |row, step| {
        let instruction = postflight.instruction(step);
        let shamt = instruction.c.as_canonical_u32() as usize;
        if shamt >= WORD_U16_LIMBS * U16_BITS {
            return Err(PostflightError::new(
                "word arithmetic shift immediate is out of range",
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
                let (output, limb_shift, bit_shift) =
                    run_shift_right_arithmetic::<WORD_U16_LIMBS, U16_BITS>(&input, &shamt_limbs);
                shifts = (limb_shift, bit_shift);
                output
            },
        )?;

        fill_core(
            &chip.inner.range_checker_chip,
            core_row.borrow_mut(),
            input,
            output,
            shifts,
        );
        Ok(())
    })?;

    Ok(trace)
}

fn fill_core<F: PrimeField32, const NUM_LIMBS: usize>(
    range_checker: &SharedVariableRangeCheckerChip,
    core_row: &mut ShiftRightArithmeticImmCoreCols<F, NUM_LIMBS, U16_BITS>,
    input: [u16; NUM_LIMBS],
    output: [u16; NUM_LIMBS],
    (limb_shift, bit_shift): (usize, usize),
) {
    let aux_bits = U16_BITS - bit_shift;
    for (limb, (carry_col, aux_col)) in input.iter().copied().zip(
        core_row
            .bit_shift_carry
            .iter_mut()
            .zip(core_row.bit_shift_aux.iter_mut()),
    ) {
        let limb = limb as u32;
        let carry = limb & ((1u32 << bit_shift) - 1);
        let aux = limb >> bit_shift;
        range_checker.add_count(carry, bit_shift);
        range_checker.add_count(aux, aux_bits);
        *carry_col = F::from_u32(carry);
        *aux_col = F::from_u32(aux);
    }

    core_row.limb_shift_marker[limb_shift] = F::ONE;
    core_row.bit_shift_marker[bit_shift] = F::ONE;
    let b_sign = input[NUM_LIMBS - 1] >> (U16_BITS - 1);
    range_checker.add_count(
        (input[NUM_LIMBS - 1] as u32) - ((b_sign as u32) << (U16_BITS - 1)),
        U16_BITS - 1,
    );
    core_row.b_sign = F::from_u16(b_sign);
    core_row.b = input.map(F::from_u16);
    core_row.a = output.map(F::from_u16);
}
