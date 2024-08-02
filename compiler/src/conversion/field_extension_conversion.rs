use p3_field::{ExtensionField, PrimeField64};

use stark_vm::cpu::OpCode::*;
use stark_vm::{cpu::trace::Instruction, field_extension::BETA};

use crate::asm::AsmInstruction;

use super::{inst, register, AS};

fn convert_field_extension_mult<const WORD_SIZE: usize, F: PrimeField64>(
    dst: i32,
    lhs: [F; 4],
    rhs: [F; 4],
    as_type: AS,
    utility_registers: [F; 8],
) -> Vec<Instruction<F>> {
    let word_size_i32: i32 = WORD_SIZE as i32;
    let beta_f = F::from_canonical_usize(BETA);

    let b0 = lhs[0];
    let b1 = lhs[1];
    let b2 = lhs[2];
    let b3 = lhs[3];

    let c0 = rhs[0];
    let c1 = rhs[1];
    let c2 = rhs[2];
    let c3 = rhs[3];

    let x0 = utility_registers[0];

    let z0 = utility_registers[1];
    let z1 = utility_registers[2];
    let z2 = utility_registers[3];
    let z3 = utility_registers[4];

    let mut instructions: Vec<Instruction<F>> = vec![];

    // This computes the constant term of the resulting polynomial:
    // z_0 = b_0 * c_0 + BETA * (b_1 * c_3 + b_2 * c_2 + b_3 * c_1)
    let a0_inst = vec![
        inst(FMUL, z0, b1, c3, AS::Register, as_type),
        inst(FMUL, x0, b2, c2, AS::Register, as_type),
        inst(FADD, z0, z0, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b3, c1, AS::Register, as_type),
        inst(FADD, z0, z0, x0, AS::Register, AS::Register),
        inst(FMUL, z0, z0, beta_f, AS::Register, AS::Immediate),
        inst(FMUL, x0, b0, c0, AS::Register, as_type),
        inst(FADD, z0, z0, x0, AS::Register, AS::Register),
    ];

    // This computes the coefficient of x in the resulting polynomial:
    // b_0 * c_1 + b_1 * c_0 + BETA * (b_2 * c_3 + b_3 * c_2)
    let a1_inst = vec![
        inst(FMUL, z1, b2, c3, AS::Register, as_type),
        inst(FMUL, x0, b3, c2, AS::Register, as_type),
        inst(FADD, z1, z1, x0, AS::Register, AS::Register),
        inst(FMUL, z1, z1, beta_f, AS::Register, AS::Immediate),
        inst(FMUL, x0, b0, c1, AS::Register, as_type),
        inst(FADD, z1, z1, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b1, c0, AS::Register, as_type),
        inst(FADD, z1, z1, x0, AS::Register, AS::Register),
    ];

    // This computes the coefficient of x^2 in the resulting polynomial:
    // b_0 * c_2 + b_1 * c_1 + b_2 * c_0 + BETA * b_3 * c_3
    let a2_inst = vec![
        inst(FMUL, z2, b3, c3, AS::Register, as_type),
        inst(FMUL, z2, z2, beta_f, AS::Register, AS::Immediate),
        inst(FMUL, x0, b0, c2, AS::Register, as_type),
        inst(FADD, z2, z2, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b1, c1, AS::Register, as_type),
        inst(FADD, z2, z2, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b2, c0, AS::Register, as_type),
        inst(FADD, z2, z2, x0, AS::Register, AS::Register),
    ];

    // This computes the coefficient of x^3 in the resulting polynomial:
    // b_0 * c_3 + b_1 * c_2 + b_2 * c_1 + b_3 * c_0
    let a3_inst = vec![
        inst(FMUL, z3, b0, c3, AS::Register, as_type),
        inst(FMUL, x0, b1, c2, AS::Register, as_type),
        inst(FADD, z3, z3, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b2, c1, AS::Register, as_type),
        inst(FADD, z3, z3, x0, AS::Register, AS::Register),
        inst(FMUL, x0, b3, c0, AS::Register, as_type),
        inst(FADD, z3, z3, x0, AS::Register, AS::Register),
    ];

    instructions.extend(a0_inst);
    instructions.extend(a1_inst);
    instructions.extend(a2_inst);
    instructions.extend(a3_inst);

    let a0 = register(dst);
    let a1 = register(dst - word_size_i32);
    let a2 = register(dst - 2 * word_size_i32);
    let a3 = register(dst - 3 * word_size_i32);

    instructions.extend([
        inst(FADD, a0, z0, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a1, z1, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a2, z2, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a3, z3, F::zero(), AS::Register, AS::Immediate),
    ]);

    instructions
}

fn convert_field_extension_inv<const WORD_SIZE: usize, F: PrimeField64>(
    dst: i32,
    src: i32,
    utility_registers: [F; 8],
) -> Vec<Instruction<F>> {
    let word_size_i32: i32 = WORD_SIZE as i32;
    let beta_f = F::from_canonical_usize(BETA);

    // for intermediate calculations
    let x0 = utility_registers[0];
    let x1 = utility_registers[1];
    let x2 = utility_registers[2];
    let x3 = utility_registers[3];

    // to store result before copying back to dst (necessary if dst == src)
    let z0 = utility_registers[4];
    let z1 = utility_registers[5];
    let z2 = utility_registers[6];
    let z3 = utility_registers[7];

    let b0 = register(src);
    let b1 = register(src - word_size_i32);
    let b2 = register(src - 2 * word_size_i32);
    let b3 = register(src - 3 * word_size_i32);

    let mut instructions = vec![];

    // First we compute the term b_0^2 - 11 * (2b_1 * b_3 - b_2^2), call this n
    let n_inst = vec![
        inst(FMUL, x0, b1, b3, AS::Register, AS::Register),
        inst(FMUL, x0, x0, F::two(), AS::Register, AS::Immediate),
        inst(FMUL, x1, b2, b2, AS::Register, AS::Register),
        inst(FSUB, x0, x0, x1, AS::Register, AS::Register),
        inst(FMUL, x0, x0, beta_f, AS::Register, AS::Immediate),
        inst(FMUL, x1, b0, b0, AS::Register, AS::Register),
        inst(FSUB, x0, x1, x0, AS::Register, AS::Register),
    ];

    // Next we compute the term 2 * b_0 * b_2 - b_1^2 - 11 * b_3^2, call this m
    let m_inst = vec![
        inst(FMUL, x1, b0, b2, AS::Register, AS::Register),
        inst(FMUL, x1, x1, F::two(), AS::Register, AS::Immediate),
        inst(FMUL, x2, b1, b1, AS::Register, AS::Register),
        inst(FSUB, x1, x1, x2, AS::Register, AS::Register),
        inst(FMUL, x2, b3, b3, AS::Register, AS::Register),
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
        inst(FMUL, z0, b0, x0, AS::Register, AS::Register),
        inst(FMUL, x2, b2, x1, AS::Register, AS::Register),
        inst(FMUL, x2, x2, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, z0, z0, x2, AS::Register, AS::Register),
    ];

    // We compute the coefficient of x: -b_1 * n + 11 * b_3 * m
    let a1_inst = vec![
        inst(FMUL, z1, b1, x0, AS::Register, AS::Register),
        inst(FMUL, x2, b3, x1, AS::Register, AS::Register),
        inst(FMUL, x2, x2, beta_f, AS::Register, AS::Immediate),
        inst(FSUB, z1, x2, z1, AS::Register, AS::Register),
    ];

    // Here, we compute the coefficient of x^2: b_2 * n - b_0 * m
    let a2_inst = vec![
        inst(FMUL, z2, b2, x0, AS::Register, AS::Register),
        inst(FMUL, x2, b0, x1, AS::Register, AS::Register),
        inst(FSUB, z2, z2, x2, AS::Register, AS::Register),
    ];

    // Finally, we compute the coefficient of x^3: b_1 * m - b_3 * n
    let a3_inst = vec![
        inst(FMUL, z3, b1, x1, AS::Register, AS::Register),
        inst(FMUL, x2, b3, x0, AS::Register, AS::Register),
        inst(FSUB, z3, z3, x2, AS::Register, AS::Register),
    ];

    instructions.extend(n_inst);
    instructions.extend(m_inst);
    instructions.extend(inv_c_inst);
    instructions.extend(mul_inst);
    instructions.extend(a0_inst);
    instructions.extend(a1_inst);
    instructions.extend(a2_inst);
    instructions.extend(a3_inst);

    let a0 = register(dst);
    let a1 = register(dst - word_size_i32);
    let a2 = register(dst - 2 * word_size_i32);
    let a3 = register(dst - 3 * word_size_i32);

    instructions.extend([
        inst(FADD, a0, z0, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a1, z1, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a2, z2, F::zero(), AS::Register, AS::Immediate),
        inst(FADD, a3, z3, F::zero(), AS::Register, AS::Immediate),
    ]);

    instructions
}

pub fn convert_field_extension_with_base<
    const WORD_SIZE: usize,
    F: PrimeField64,
    EF: ExtensionField<F>,
>(
    instruction: AsmInstruction<F, EF>,
    utility_registers: [F; 8],
) -> Vec<Instruction<F>> {
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

            vec![
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

            vec![
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
            ]
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
                utility_registers,
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
                utility_registers,
            )
        }
        AsmInstruction::InvE(dst, src) => {
            convert_field_extension_inv::<WORD_SIZE, F>(dst, src, utility_registers)
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
        AsmInstruction::MulEI(_, _, _) => {
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
        AsmInstruction::InvE(dst, src) => vec![inst(
            BBE4INV,
            register(dst),
            register(src),
            register(src),
            AS::Register,
            AS::Register,
        )],
        _ => panic!(
            "Illegal argument to convert_field_extension: {:?}",
            instruction
        ),
    }
}
