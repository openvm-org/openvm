use p3_field::PrimeField64;

use crate::asm::{AsmInstruction, AssemblyCode};

use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::OpCode;
use stark_vm::cpu::OpCode::*;

fn inst<F: PrimeField64>(
    opcode: OpCode,
    op_a: F,
    op_b: F,
    op_c: F,
    d: AS,
    e: AS,
) -> Instruction<F> {
    Instruction {
        opcode,
        op_a,
        op_b,
        op_c,
        d: d.to_field(),
        e: e.to_field(),
    }
}

enum AS {
    Immediate,
    Register,
    Memory,
}

impl AS {
    fn to_field<F: PrimeField64>(&self) -> F {
        match self {
            AS::Immediate => F::zero(),
            AS::Register => F::one(),
            AS::Memory => F::two(),
        }
    }
}

fn register<F: PrimeField64>(value: i32) -> F {
    let value = -value;
    //println!("register index: {}", value);
    assert!(value >= 0);
    F::from_canonical_usize(value as usize)
}

const UTILITY_REGISTER: i32 = 0;

fn convert_instruction<F: PrimeField64>(
    instruction: AsmInstruction<F, F>,
    pc: F,
    labels: impl Fn(F) -> F,
) -> Vec<Instruction<F>> {
    let utility_register = register(UTILITY_REGISTER);
    match instruction {
        AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
        AsmInstruction::LoadF(dst, src, index, offset, size) => (0..size.as_canonical_u64())
            .flat_map(|i| {
                vec![
                    // register[util] <- register[src] + register[index]
                    inst(
                        FADD,
                        utility_register,
                        register(src),
                        register(index),
                        AS::Register,
                        AS::Register,
                    ),
                    // register[dst] <- mem[register[util] + offset]
                    inst(
                        LOADW,
                        register(dst),
                        offset + F::from_canonical_u64(i),
                        utility_register,
                        AS::Register,
                        AS::Memory,
                    ),
                ]
            })
            .collect(),
        AsmInstruction::LoadFI(dst, src, index, offset, size) => (0..size.as_canonical_u64())
            .flat_map(|i| {
                vec![
                    // register[dst] <- mem[register[src] + (index + offset)]
                    inst(
                        LOADW,
                        register(dst),
                        index + offset + F::from_canonical_u64(i),
                        register(src),
                        AS::Register,
                        AS::Memory,
                    ),
                ]
            })
            .collect(),
        AsmInstruction::StoreF(val, addr, index, offset, size) => (0..size.as_canonical_u64())
            .flat_map(|i| {
                vec![
                    // register[util] <- register[addr] + register[index]
                    inst(
                        FADD,
                        utility_register,
                        register(addr),
                        register(index),
                        AS::Register,
                        AS::Register,
                    ),
                    //  mem[register[util] + offset] <- register[val]
                    inst(
                        STOREW,
                        register(val),
                        offset + F::from_canonical_u64(i),
                        utility_register,
                        AS::Register,
                        AS::Memory,
                    ),
                ]
            })
            .collect(),
        AsmInstruction::StoreFI(val, addr, index, offset, size) => (0..size.as_canonical_u64())
            .flat_map(|i| {
                vec![
                    // mem[register[addr] + (index + offset)] <- register[val]
                    inst(
                        STOREW,
                        register(val),
                        index + offset + F::from_canonical_u64(i),
                        register(addr),
                        AS::Register,
                        AS::Memory,
                    ),
                ]
            })
            .collect(),
        AsmInstruction::AddF(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] + register[rhs]
            inst(
                FADD,
                register(dst),
                register(lhs),
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::AddFI(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] + rhs
            inst(
                FADD,
                register(dst),
                register(lhs),
                rhs,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::SubF(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] - register[rhs]
            inst(
                FSUB,
                register(dst),
                register(lhs),
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::SubFI(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] - rhs
            inst(
                FSUB,
                register(dst),
                register(lhs),
                rhs,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::SubFIN(dst, lhs, rhs) => vec![
            // register[util] <- lhs
            inst(
                STOREW,
                lhs,
                F::zero(),
                register(rhs),
                AS::Immediate,
                AS::Register,
            ),
            // register[dst] <- register[util] - register[rhs]
            inst(
                FSUB,
                register(dst),
                utility_register,
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::MulF(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] * register[rhs]
            inst(
                FMUL,
                register(dst),
                register(lhs),
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::MulFI(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] * rhs
            inst(
                FMUL,
                register(dst),
                register(lhs),
                rhs,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::DivF(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] / register[rhs]
            inst(
                FDIV,
                register(dst),
                register(lhs),
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::DivFI(dst, lhs, rhs) => vec![
            // register[dst] <- register[lhs] / rhs
            inst(
                FDIV,
                register(dst),
                register(lhs),
                rhs,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::DivFIN(dst, lhs, rhs) => vec![
            // register[util] <- lhs
            inst(
                STOREW,
                lhs,
                F::zero(),
                register(rhs),
                AS::Immediate,
                AS::Register,
            ),
            // register[dst] <- register[util] / register[rhs]
            inst(
                FDIV,
                register(dst),
                utility_register,
                register(rhs),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::Jal(dst, label, offset) => vec![
            // pc <- labels[label] + offset, register[dst] <- pc
            inst(
                JAL,
                register(dst),
                labels(label) + offset - pc,
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::Bne(label, lhs, rhs) => vec![
            // if register[lhs] != register[rhs], pc <- labels[label]
            inst(
                BNE,
                register(lhs),
                register(rhs),
                labels(label) - pc,
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::BneInc(label, lhs, rhs) => vec![
            // register[lhs] += 1
            inst(
                FADD,
                register(lhs),
                register(lhs),
                F::one(),
                AS::Register,
                AS::Immediate,
            ),
            // if register[lhs] != register[rhs], pc <- labels[label]
            inst(
                BNE,
                register(lhs),
                register(rhs),
                labels(label) - (pc + F::one()),
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::BneI(label, lhs, rhs) => vec![
            // if register[lhs] != rhs, pc <- labels[label]
            inst(
                BNE,
                register(lhs),
                rhs,
                labels(label) - pc,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::BneIInc(label, lhs, rhs) => vec![
            // register[lhs] += 1
            inst(
                FADD,
                register(lhs),
                register(lhs),
                F::one(),
                AS::Register,
                AS::Immediate,
            ),
            // if register[lhs] != rhs, pc <- labels[label]
            inst(
                BNE,
                register(lhs),
                rhs,
                labels(label) - (pc + F::one()),
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::Beq(label, lhs, rhs) => vec![
            // if register[lhs] == register[rhs], pc <- labels[label]
            inst(
                BEQ,
                register(lhs),
                register(rhs),
                labels(label) - pc,
                AS::Register,
                AS::Register,
            ),
        ],
        AsmInstruction::BeqI(label, lhs, rhs) => vec![
            // if register[lhs] == rhs, pc <- labels[label]
            inst(
                BEQ,
                register(lhs),
                rhs,
                labels(label) - pc,
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::Trap => vec![
            // pc <- -1 (causes trace generation to fail)
            inst(
                JAL,
                utility_register,
                -pc - F::one(),
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
        ],
        AsmInstruction::Halt => vec![
            // terminate
            inst(
                TERMINATE,
                F::zero(),
                F::zero(),
                F::zero(),
                AS::Immediate,
                AS::Immediate,
            ),
        ],
        _ => panic!("Unsupported instruction {:?}", instruction),
    }
}

pub fn convert_program<F: PrimeField64>(program: AssemblyCode<F, F>) -> Vec<Instruction<F>> {
    let mut block_start = vec![];
    let mut pc = 0;
    for block in program.blocks.iter() {
        block_start.push(pc);
        for instruction in block.0.iter() {
            let instructions =
                convert_instruction(instruction.clone(), F::from_canonical_usize(pc), |label| {
                    label
                });
            pc += instructions.len();
        }
    }

    println!("{:?}", block_start);

    let mut result = vec![];
    for block in program.blocks.iter() {
        for instruction in block.0.iter() {
            let labels =
                |label: F| F::from_canonical_usize(block_start[label.as_canonical_u64() as usize]);
            result.extend(convert_instruction(
                instruction.clone(),
                F::from_canonical_usize(result.len()),
                labels,
            ));
        }
    }

    result
}
