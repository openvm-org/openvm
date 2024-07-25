use p3_field::{ExtensionField, PrimeField64};

use crate::asm::AsmInstruction;

use stark_vm::cpu::OpCode::*;
use stark_vm::{cpu::trace::Instruction, field_extension::BETA};

use super::{inst, register, AS};

fn convert_field_extension_mult<const WORD_SIZE: usize, F: PrimeField64>(
    dst: i32,
    lhs_registers: [F; 4],
    rhs: [F; 4],
    as_type: AS,
    x0: F,
) -> Vec<Instruction<F>> {
    let word_size_i32: i32 = WORD_SIZE as i32;
    let beta_f = F::from_canonical_usize(BETA);

    let a0 = dst;
    let a1 = dst - word_size_i32;
    let a2 = dst - 2 * word_size_i32;
    let a3 = dst - 3 * word_size_i32;

    let b0 = lhs_registers[0];
    let b1 = lhs_registers[1];
    let b2 = lhs_registers[2];
    let b3 = lhs_registers[3];

    let c0 = rhs[0];
    let c1 = rhs[1];
    let c2 = rhs[2];
    let c3 = rhs[3];

    let mut instructions: Vec<Instruction<F>> = vec![];

    // This computes the constant term of the resulting polynomial:
    // a_0 = b_0 * c_0 + BETA * (b_1 * c_3 + b_2 * c_2 + b_3 * c_1)
    let a0_inst = vec![
        inst(FMUL, register(a0), b1, c3, AS::Register, as_type),
        inst(FMUL, x0, b2, c2, AS::Register, as_type),
        inst(
            FADD,
            register(a0),
            register(a0),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b3, c1, AS::Register, as_type),
        inst(
            FADD,
            register(a0),
            register(a0),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(
            FMUL,
            register(a0),
            register(a0),
            beta_f,
            AS::Register,
            AS::Immediate,
        ),
        inst(FMUL, x0, b0, c0, AS::Register, as_type),
        inst(
            FADD,
            register(a0),
            register(a0),
            x0,
            AS::Register,
            AS::Register,
        ),
    ];

    // This computes the coefficient of x in the resulting polynomial:
    // b_0 * c_1 + b_1 * c_0 + BETA * (b_2 * c_3 + b_3 * c_2)
    let a1_inst = vec![
        inst(FMUL, register(a1), b2, c3, AS::Register, as_type),
        inst(FMUL, x0, b3, c2, AS::Register, as_type),
        inst(
            FADD,
            register(a1),
            register(a1),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(
            FMUL,
            register(a1),
            register(a1),
            beta_f,
            AS::Register,
            AS::Immediate,
        ),
        inst(FMUL, x0, b0, c1, AS::Register, as_type),
        inst(
            FADD,
            register(a1),
            register(a1),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b1, c0, AS::Register, as_type),
        inst(
            FADD,
            register(a1),
            register(a1),
            x0,
            AS::Register,
            AS::Register,
        ),
    ];

    // This computes the coefficient of x^2 in the resulting polynomial:
    // b_0 * c_2 + b_1 * c_1 + b_2 * c_0 + BETA * b_3 * c_3
    let a2_inst = vec![
        inst(FMUL, register(a2), b3, c3, AS::Register, as_type),
        inst(
            FMUL,
            register(a2),
            register(a2),
            beta_f,
            AS::Register,
            AS::Immediate,
        ),
        inst(FMUL, x0, b0, c2, AS::Register, as_type),
        inst(
            FADD,
            register(a2),
            register(a2),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b1, c1, AS::Register, as_type),
        inst(
            FADD,
            register(a2),
            register(a2),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b2, c0, AS::Register, as_type),
        inst(
            FADD,
            register(a2),
            register(a2),
            x0,
            AS::Register,
            AS::Register,
        ),
    ];

    // This computes the coefficient of x^3 in the resulting polynomial:
    // b_0 * c_3 + b_1 * c_2 + b_2 * c_1 + b_3 * c_0
    let a3_inst = vec![
        inst(FMUL, register(a3), b0, c3, AS::Register, as_type),
        inst(FMUL, x0, b1, c2, AS::Register, as_type),
        inst(
            FADD,
            register(a3),
            register(a3),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b2, c1, AS::Register, as_type),
        inst(
            FADD,
            register(a3),
            register(a3),
            x0,
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, b3, c0, AS::Register, as_type),
        inst(
            FADD,
            register(a3),
            register(a3),
            x0,
            AS::Register,
            AS::Register,
        ),
    ];

    instructions.extend(a0_inst);
    instructions.extend(a1_inst);
    instructions.extend(a2_inst);
    instructions.extend(a3_inst);

    instructions
}

fn convert_field_extension_inv<const WORD_SIZE: usize, F: PrimeField64>(
    dst_registers: [F; 4],
    src: i32,
    utility_registers: [F; 4],
) -> Vec<Instruction<F>> {
    let word_size_i32: i32 = WORD_SIZE as i32;
    let beta_f = F::from_canonical_usize(BETA);

    let x0 = utility_registers[0];
    let x1 = utility_registers[1];
    let x2 = utility_registers[2];
    let x3 = utility_registers[3];

    let a0 = dst_registers[0];
    let a1 = dst_registers[1];
    let a2 = dst_registers[2];
    let a3 = dst_registers[3];

    let b0 = src;
    let b1 = src - word_size_i32;
    let b2 = src - 2 * word_size_i32;
    let b3 = src - 3 * word_size_i32;

    let mut instructions = vec![];

    // First we compute the term b_0^2 - 11 * (2b_1 * b_3 - b_2^2), call this n
    let n_inst = vec![
        inst(
            FMUL,
            x0,
            register(b1),
            register(b3),
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x0, x0, F::two(), AS::Register, AS::Immediate),
        inst(
            FMUL,
            x1,
            register(b2),
            register(b2),
            AS::Register,
            AS::Register,
        ),
        inst(FSUB, x0, x0, x1, AS::Register, AS::Register),
        inst(FMUL, x0, x0, beta_f, AS::Register, AS::Immediate),
        inst(
            FMUL,
            x1,
            register(b0),
            register(b0),
            AS::Register,
            AS::Register,
        ),
        inst(FSUB, x0, x1, x0, AS::Register, AS::Register),
    ];

    // Next we compute the term 2 * b_0 * b_2 - b_1^2 - 11 * b_3^2, call this m
    let m_inst = vec![
        inst(
            FMUL,
            x1,
            register(b0),
            register(b2),
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x1, x1, F::two(), AS::Register, AS::Immediate),
        inst(
            FMUL,
            x2,
            register(b1),
            register(b1),
            AS::Register,
            AS::Register,
        ),
        inst(FSUB, x1, x1, x2, AS::Register, AS::Register),
        inst(
            FMUL,
            x2,
            register(b3),
            register(b3),
            AS::Register,
            AS::Register,
        ),
        inst(FMUL, x2, x2, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, x1, x1, x2, AS::Register, AS::Register),
    ];

    // Now, we compute the term c = n^2 - 11*m^2, and then take the inverse, call this inv_c
    let inv_c_inst = vec![
        inst(FMUL, x2, x0, x0, AS::Register, AS::Register),
        inst(FMUL, x3, x1, x1, AS::Register, AS::Register),
        inst(FMUL, x3, x3, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, x2, x2, x3, AS::Register, AS::Register),
        inst(STOREW, F::one(), F::zero(), x3, AS::Immediate, AS::Register),
        inst(FDIV, x2, x3, x2, AS::Register, AS::Register),
    ];

    // Now, we multiply n and m by inv_c
    let mul_inst = vec![
        inst(FMUL, x0, x0, x2, AS::Register, AS::Register),
        inst(FMUL, x1, x1, x2, AS::Register, AS::Register),
    ];

    // We compute the constant term of the result: b_0 * n - 11 * b_2 * m
    let a0_inst = vec![
        inst(FMUL, a0, register(b0), x0, AS::Register, AS::Register),
        inst(FMUL, x2, register(b2), x1, AS::Register, AS::Register),
        inst(FMUL, x2, x2, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, a0, a0, x2, AS::Register, AS::Register),
    ];

    // We compute the coefficient of x: -b_1 * n + 11 * b_3 * m
    let a1_inst = vec![
        inst(FMUL, a1, register(b1), x0, AS::Register, AS::Register),
        inst(FMUL, x2, register(b3), x1, AS::Register, AS::Register),
        inst(FMUL, x2, x2, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, a1, x2, a1, AS::Register, AS::Register),
    ];

    // Here, we compute the coefficient of x^2: b_2 * n - b_0 * m
    let a2_inst = vec![
        inst(FMUL, a2, register(b2), x0, AS::Register, AS::Register),
        inst(FMUL, x2, register(b0), x1, AS::Register, AS::Register),
        inst(FSUB, a2, a2, x2, AS::Register, AS::Register),
    ];

    // Finally, we compute the coefficient of x^3: b_1 * m - b_3 * n
    let a3_inst = vec![
        inst(FMUL, a3, register(b1), x1, AS::Register, AS::Register),
        inst(FMUL, x2, register(b3), x0, AS::Register, AS::Register),
        inst(FSUB, a3, a3, x2, AS::Register, AS::Register),
    ];

    instructions.extend(n_inst);
    instructions.extend(m_inst);
    instructions.extend(inv_c_inst);
    instructions.extend(mul_inst);
    instructions.extend(a0_inst);
    instructions.extend(a1_inst);
    instructions.extend(a2_inst);
    instructions.extend(a3_inst);

    instructions
}

fn field_extension_inv_immediate<F: PrimeField64, EF: ExtensionField<F>>(x: EF) -> [F; 4] {
    let slc = x.as_base_slice();
    let x0 = slc[0];
    let x1 = slc[1];
    let x2 = slc[2];
    let x3 = slc[3];

    let beta_f = F::from_canonical_usize(BETA);

    // First we compute the term x_0^2 - 11 * (2x_1 * x_3 - x_2^2), call this n
    let n = x0 * x0 - beta_f * (F::two() * x1 * x3 - x2 * x2);

    // Next we compute the term 2 * x_0 * x_2 - x_1^2 - 11 * x_3^2, call this m
    let m = F::two() * x0 * x2 - x1 * x1 - beta_f * x3 * x3;

    // Now, we compute the term c = n^2 - 11 * m^2, and then take the inverse, call this inv_c
    let inv_c = n * n - beta_f * m * m;

    // Now, we multiply x_0 and x_1 by inv_c
    let n = n * inv_c;
    let m = m * inv_c;

    // We compute the constant term of the result: x_0 * n - 11 * x_2 * m
    let a0 = x0 * n - beta_f * x2 * m;

    // We compute the coefficient of x: -x_1 * n + 11 * x_3 * m
    let a1 = -x1 * n + beta_f * x3 * m;

    // Here, we compute the coefficient of x^2: x_2 * n - x_0 * m
    let a2 = x2 * n - x0 * m;

    // Finally, we compute the coefficient of x^3: x_1 * m - x_3 * n
    let a3 = x1 * m - x3 * n;

    [a0, a1, a2, a3]
}

pub fn convert_field_extension_with_base<
    const WORD_SIZE: usize,
    F: PrimeField64,
    EF: ExtensionField<F>,
>(
    instruction: AsmInstruction<F, EF>,
    utility_registers: [F; 8],
) -> Vec<Instruction<F>> {
    let x0 = utility_registers[0];
    let x1 = utility_registers[1];
    let x2 = utility_registers[2];
    let x3 = utility_registers[3];

    let main_utility = [x0, x1, x2, x3];

    let div_utility = [
        utility_registers[4],
        utility_registers[5],
        utility_registers[6],
        utility_registers[7],
    ];

    let word_size_i32: i32 = WORD_SIZE as i32;

    match instruction {
        AsmInstruction::AddE(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let b0 = lhs;
            let b1 = lhs - word_size_i32;
            let b2 = lhs - 2 * word_size_i32;
            let b3 = lhs - 3 * word_size_i32;

            let c0 = rhs;
            let c1 = rhs - word_size_i32;
            let c2 = rhs - 2 * word_size_i32;
            let c3 = rhs - 3 * word_size_i32;

            let instructions = vec![
                inst(
                    FADD,
                    register(a0),
                    register(b0),
                    register(c0),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FADD,
                    register(a1),
                    register(b1),
                    register(c1),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FADD,
                    register(a2),
                    register(b2),
                    register(c2),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FADD,
                    register(a3),
                    register(b3),
                    register(c3),
                    AS::Register,
                    AS::Register,
                ),
            ];

            instructions
        }
        AsmInstruction::AddEI(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let b0 = lhs;
            let b1 = lhs - word_size_i32;
            let b2 = lhs - 2 * word_size_i32;
            let b3 = lhs - 3 * word_size_i32;

            let slc = rhs.as_base_slice();
            let c0 = slc[0];
            let c1 = slc[1];
            let c2 = slc[2];
            let c3 = slc[3];

            let instructions = vec![
                inst(
                    FADD,
                    register(a0),
                    register(b0),
                    c0,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FADD,
                    register(a1),
                    register(b1),
                    c1,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FADD,
                    register(a2),
                    register(b2),
                    c2,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FADD,
                    register(a3),
                    register(b3),
                    c3,
                    AS::Register,
                    AS::Immediate,
                ),
            ];

            instructions
        }
        AsmInstruction::AddEFFI(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let slc = rhs.as_base_slice();
            let c0 = slc[0];
            let c1 = slc[1];
            let c2 = slc[2];
            let c3 = slc[3];

            vec![
                inst(
                    FADD,
                    register(a0),
                    register(lhs),
                    c0,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    STOREW,
                    c1,
                    F::zero(),
                    register(a1),
                    AS::Immediate,
                    AS::Register,
                ),
                inst(
                    STOREW,
                    c2,
                    F::zero(),
                    register(a2),
                    AS::Immediate,
                    AS::Register,
                ),
                inst(
                    STOREW,
                    c3,
                    F::zero(),
                    register(a3),
                    AS::Immediate,
                    AS::Register,
                ),
            ]
        }
        AsmInstruction::SubE(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let b0 = lhs;
            let b1 = lhs - word_size_i32;
            let b2 = lhs - 2 * word_size_i32;
            let b3 = lhs - 3 * word_size_i32;

            let c0 = rhs;
            let c1 = rhs - word_size_i32;
            let c2 = rhs - 2 * word_size_i32;
            let c3 = rhs - 3 * word_size_i32;

            let instructions = vec![
                inst(
                    FSUB,
                    register(a0),
                    register(b0),
                    register(c0),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a1),
                    register(b1),
                    register(c1),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a2),
                    register(b2),
                    register(c2),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a3),
                    register(b3),
                    register(c3),
                    AS::Register,
                    AS::Register,
                ),
            ];

            instructions
        }
        AsmInstruction::SubEI(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let b0 = lhs;
            let b1 = lhs - word_size_i32;
            let b2 = lhs - 2 * word_size_i32;
            let b3 = lhs - 3 * word_size_i32;

            let slc = rhs.as_base_slice();
            let c0 = slc[0];
            let c1 = slc[1];
            let c2 = slc[2];
            let c3 = slc[3];

            let instructions = vec![
                inst(
                    FSUB,
                    register(a0),
                    register(b0),
                    c0,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FSUB,
                    register(a1),
                    register(b1),
                    c1,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FSUB,
                    register(a2),
                    register(b2),
                    c2,
                    AS::Register,
                    AS::Immediate,
                ),
                inst(
                    FSUB,
                    register(a3),
                    register(b3),
                    c3,
                    AS::Register,
                    AS::Immediate,
                ),
            ];

            instructions
        }
        AsmInstruction::SubEIN(dst, lhs, rhs) => {
            let a0 = dst;
            let a1 = dst - word_size_i32;
            let a2 = dst - 2 * word_size_i32;
            let a3 = dst - 3 * word_size_i32;

            let slc = lhs.as_base_slice();
            let b0 = slc[0];
            let b1 = slc[1];
            let b2 = slc[2];
            let b3 = slc[3];

            let c0 = rhs;
            let c1 = rhs - word_size_i32;
            let c2 = rhs - 2 * word_size_i32;
            let c3 = rhs - 3 * word_size_i32;

            let instructions = vec![
                inst(STOREW, b0, F::zero(), x0, AS::Immediate, AS::Register),
                inst(STOREW, b1, F::zero(), x1, AS::Immediate, AS::Register),
                inst(STOREW, b2, F::zero(), x2, AS::Immediate, AS::Register),
                inst(STOREW, b3, F::zero(), x3, AS::Immediate, AS::Register),
                inst(
                    FSUB,
                    register(a0),
                    x0,
                    register(c0),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a1),
                    x1,
                    register(c1),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a2),
                    x2,
                    register(c2),
                    AS::Register,
                    AS::Register,
                ),
                inst(
                    FSUB,
                    register(a3),
                    x3,
                    register(c3),
                    AS::Register,
                    AS::Register,
                ),
            ];

            instructions
        }
        AsmInstruction::MulE(dst, lhs, rhs) => {
            let rhs_register = [
                register(rhs),
                register(rhs - word_size_i32),
                register(rhs - 2 * word_size_i32),
                register(rhs - 3 * word_size_i32),
            ];
            let lhs_register = [
                register(lhs),
                register(lhs - word_size_i32),
                register(lhs - 2 * word_size_i32),
                register(lhs - 3 * word_size_i32),
            ];
            convert_field_extension_mult::<WORD_SIZE, F>(
                dst,
                lhs_register,
                rhs_register,
                AS::Register,
                x0,
            )
        }
        AsmInstruction::MulEI(dst, lhs, rhs) => {
            let lhs_register = [
                register(lhs),
                register(lhs - word_size_i32),
                register(lhs - 2 * word_size_i32),
                register(lhs - 3 * word_size_i32),
            ];
            let rhs_slc = rhs.as_base_slice().try_into().unwrap();
            convert_field_extension_mult::<WORD_SIZE, F>(
                dst,
                lhs_register,
                rhs_slc,
                AS::Immediate,
                x0,
            )
        }
        AsmInstruction::DivE(dst, lhs, rhs) => {
            let inv_instr =
                convert_field_extension_inv::<WORD_SIZE, F>(div_utility, rhs, main_utility);

            let lhs_register = [
                register(lhs),
                register(lhs - word_size_i32),
                register(lhs - 2 * word_size_i32),
                register(lhs - 3 * word_size_i32),
            ];

            let mul_instr = convert_field_extension_mult::<WORD_SIZE, F>(
                dst,
                lhs_register,
                div_utility,
                AS::Register,
                x0,
            );

            inv_instr.into_iter().chain(mul_instr).collect()
        }
        AsmInstruction::DivEI(dst, lhs, rhs) => {
            let lhs_register = [
                register(lhs),
                register(lhs - word_size_i32),
                register(lhs - 2 * word_size_i32),
                register(lhs - 3 * word_size_i32),
            ];
            let rhs_inv = field_extension_inv_immediate::<F, EF>(rhs);

            convert_field_extension_mult::<WORD_SIZE, F>(
                dst,
                lhs_register,
                rhs_inv,
                AS::Immediate,
                x0,
            )
        }
        AsmInstruction::DivEIN(dst, lhs, rhs) => {
            let inv_instr =
                convert_field_extension_inv::<WORD_SIZE, F>(div_utility, rhs, main_utility);

            let lhs_slc = lhs.as_base_slice().try_into().unwrap();
            let mul_instr = convert_field_extension_mult::<WORD_SIZE, F>(
                dst,
                div_utility,
                lhs_slc,
                AS::Immediate,
                x0,
            );
            inv_instr.into_iter().chain(mul_instr).collect()
        }
        _ => panic!(
            "Illegal argument to convert_field_extension_with_base: {:?}",
            instruction
        ),
    }
}

pub fn convert_field_extension<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    utility_registers: [F; 8],
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::AddE(dst, lhs, rhs) => vec![inst(
            FE4ADD,
            register(dst),
            register(lhs),
            register(rhs),
            AS::Register,
            AS::Register,
        )],
        AsmInstruction::AddEI(_, _, _)
        | AsmInstruction::SubEI(_, _, _)
        | AsmInstruction::SubEIN(_, _, _)
        | AsmInstruction::MulEI(_, _, _)
        | AsmInstruction::DivEI(_, _, _)
        | AsmInstruction::DivEIN(_, _, _)
        | AsmInstruction::AddEFFI(_, _, _) => {
            convert_field_extension_with_base::<WORD_SIZE, F, EF>(instruction, utility_registers)
        }
        AsmInstruction::SubE(dst, lhs, rhs) => vec![inst(
            FE4SUB,
            register(dst),
            register(lhs),
            register(rhs),
            AS::Register,
            AS::Register,
        )],
        AsmInstruction::MulE(dst, lhs, rhs) => vec![inst(
            BBE4MUL,
            register(dst),
            register(lhs),
            register(rhs),
            AS::Register,
            AS::Register,
        )],
        AsmInstruction::DivE(dst, lhs, rhs) => vec![
            inst(
                BBE4INV,
                utility_registers[0],
                register(rhs),
                register(rhs),
                AS::Register,
                AS::Register,
            ),
            inst(
                BBE4MUL,
                register(dst),
                register(lhs),
                utility_registers[0],
                AS::Register,
                AS::Register,
            ),
        ],
        _ => panic!(
            "Illegal argument to convert_field_extension: {:?}",
            instruction
        ),
    }
}
