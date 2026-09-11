use std::{borrow::BorrowMut, iter::zip};

use openvm_circuit::{
    arch::{fill_trace_rows, Postflight, PostflightError},
    utils::next_power_of_two_or_zero,
};
use openvm_instructions::{riscv::REGISTER_NUM_LIMBS, LocalOpcode};
use openvm_riscv_transpiler::{BaseAluImmOpcode, BaseAluOpcode};
use openvm_stark_backend::{p3_field::PrimeField32, p3_matrix::dense::RowMajorMatrix};

use super::{BitwiseLogicImmChip, BitwiseLogicImmCoreCols, BYTE_BITS};
use crate::{
    adapters::{imm_to_bytes, BaseAluImmAdapterCols, BaseAluImmAdapterFiller},
    bitwise_logic::run_bitwise_logic,
};

/// Generates the RV64 immediate bitwise trace directly from immutable preflight history.
pub fn generate_trace_from_postflight<F: PrimeField32>(
    chip: &BitwiseLogicImmChip<F>,
    postflight: &Postflight<'_, F>,
) -> Result<RowMajorMatrix<F>, PostflightError> {
    let opcodes = [
        BaseAluImmOpcode::XORI,
        BaseAluImmOpcode::ORI,
        BaseAluImmOpcode::ANDI,
    ];
    let rows_used = opcodes
        .iter()
        .map(|opcode| postflight.steps(opcode.global_opcode()).len())
        .sum();
    let adapter_width = BaseAluImmAdapterCols::<F>::width();
    let width =
        adapter_width + BitwiseLogicImmCoreCols::<F, REGISTER_NUM_LIMBS, BYTE_BITS>::width();
    let height = next_power_of_two_or_zero(rows_used);
    let mut trace = RowMajorMatrix::new(F::zero_vec(height * width), width);

    let mut row_index = 0;
    for local_opcode in opcodes {
        let bitwise_opcode = match local_opcode {
            BaseAluImmOpcode::XORI => BaseAluOpcode::XOR,
            BaseAluImmOpcode::ORI => BaseAluOpcode::OR,
            BaseAluImmOpcode::ANDI => BaseAluOpcode::AND,
            BaseAluImmOpcode::ADDI => unreachable!(),
        };
        let steps = postflight.steps(local_opcode.global_opcode());
        fill_trace_rows(&mut trace, row_index, steps, |row, step| {
            let (adapter_row, core_row) = row.split_at_mut(adapter_width);
            let immediate = postflight.instruction(step).c.as_u32();
            let c = imm_to_bytes(immediate);
            let (b, a) = BaseAluImmAdapterFiller::replay(
                postflight,
                step,
                &chip.mem_helper.as_borrowed(),
                adapter_row.borrow_mut(),
                |b, _| run_bitwise_logic::<REGISTER_NUM_LIMBS, BYTE_BITS>(bitwise_opcode, &b, &c),
            )?;
            let core_row: &mut BitwiseLogicImmCoreCols<F, REGISTER_NUM_LIMBS, BYTE_BITS> =
                core_row.borrow_mut();
            let c_low = [c[0], c[1] & 0x07];
            let imm_sign = c[2] != 0;

            chip.inner
                .bitwise_lookup_chip
                .request_range(c_low[0] as u32, (c_low[1] + 0xf8) as u32);
            for (&b, &c) in zip(&b, &c) {
                chip.inner
                    .bitwise_lookup_chip
                    .request_xor(b as u32, c as u32);
            }

            core_row.opcode_and_flag = F::from_bool(local_opcode == BaseAluImmOpcode::ANDI);
            core_row.opcode_or_flag = F::from_bool(local_opcode == BaseAluImmOpcode::ORI);
            core_row.opcode_xor_flag = F::from_bool(local_opcode == BaseAluImmOpcode::XORI);
            core_row.imm_sign = F::from_bool(imm_sign);
            core_row.c_low = c_low.map(F::from_u8);
            core_row.b = b.map(F::from_u8);
            core_row.a = a.map(F::from_u8);
            Ok(())
        })?;
        row_index += steps.len();
    }

    Ok(trace)
}
