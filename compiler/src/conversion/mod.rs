use std::array::from_fn;

use p3_field::{ExtensionField, PrimeField64};

use field_extension_conversion::{convert_field_extension, convert_field_extension_with_base};
use stark_vm::cpu::trace::Instruction;
use stark_vm::cpu::OpCode;
use stark_vm::cpu::OpCode::*;

use crate::asm::{AsmInstruction, AssemblyCode};

pub mod field_extension_conversion;

#[derive(Clone, Copy)]
pub struct CompilerOptions {
    pub compile_prints: bool,
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
}

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

#[derive(Clone, Copy)]
enum AS {
    Immediate,
    Register,
    Memory,
}

impl AS {
    // TODO[INT-1698]
    fn to_field<F: PrimeField64>(self) -> F {
        match self {
            AS::Immediate => F::zero(),
            AS::Register => F::one(),
            AS::Memory => F::two(),
        }
    }
}

const POSEIDON2_WIDTH: usize = 16;
const NUM_UTILITY_REGISTERS: usize = POSEIDON2_WIDTH;

fn register<F: PrimeField64>(value: i32) -> F {
    let value = (NUM_UTILITY_REGISTERS as i32) - value;
    //println!("register index: {}", value);
    assert!(value > 0);
    F::from_canonical_usize(value as usize)
}

fn convert_base_arithmetic_instruction<F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    utility_register: F,
) -> Vec<Instruction<F>> {
    match instruction {
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
            // register[dst] <- register[rhs] - lhs
            inst(
                FSUB,
                register(dst),
                register(rhs),
                lhs,
                AS::Register,
                AS::Immediate,
            ),
            // register[dst] <- register[dst] * -1
            inst(
                FMUL,
                register(dst),
                register(dst),
                F::neg_one(),
                AS::Register,
                AS::Immediate,
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
                utility_register,
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
        _ => panic!(
            "Illegal argument to convert_field_arithmetic_instruction: {:?}",
            instruction
        ),
    }
}

fn convert_print_instruction<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
) -> Vec<Instruction<F>> {
    let word_size_i32 = WORD_SIZE as i32;

    match instruction {
        AsmInstruction::PrintV(src) => vec![inst(
            PRINTF,
            register(src),
            F::zero(),
            F::zero(),
            AS::Register,
            AS::Immediate,
        )],
        AsmInstruction::PrintF(src) => vec![inst(
            PRINTF,
            register(src),
            F::zero(),
            F::zero(),
            AS::Register,
            AS::Immediate,
        )],
        AsmInstruction::PrintE(src) => vec![
            inst(
                PRINTF,
                register(src),
                F::zero(),
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                register(src - word_size_i32),
                F::zero(),
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                register(src - 2 * word_size_i32),
                F::zero(),
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                register(src - 3 * word_size_i32),
                F::zero(),
                F::zero(),
                AS::Register,
                AS::Immediate,
            ),
        ],
        _ => panic!(
            "Illegal argument to convert_print_instruction: {:?}",
            instruction
        ),
    }
}

fn convert_instruction<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    pc: F,
    labels: impl Fn(F) -> F,
    options: CompilerOptions,
) -> Vec<Instruction<F>> {
    let utility_registers: [F; NUM_UTILITY_REGISTERS] = from_fn(|i| F::from_canonical_usize(i));
    let utility_register = utility_registers[0];

    match instruction {
        AsmInstruction::ImmE(dst, val) => {
            let val_slice = val.as_base_slice();

            (0..EF::D)
                .map(|i|
                // register[dst + i * WORD_SIZE] <- val_slice[i]
                inst(
                    STOREW,
                    val_slice[i],
                    register(dst - (i * WORD_SIZE) as i32),
                    F::zero(),
                    AS::Immediate,
                    AS::Register,
                ))
                .collect()
        }
        AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
        AsmInstruction::LoadF(dst, src, index, offset, size) => vec![
            // register[util] <- register[index] * size
            inst(
                FMUL,
                utility_register,
                register(index),
                size,
                AS::Register,
                AS::Immediate,
            ),
            // register[util] <- register[src] + register[util]
            inst(
                FADD,
                utility_register,
                register(src),
                utility_register,
                AS::Register,
                AS::Register,
            ),
            // register[dst] <- mem[register[util] + offset]
            inst(
                LOADW,
                register(dst),
                offset,
                utility_register,
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::LoadFI(dst, src, index, offset, size) => vec![
            // register[dst] <- mem[register[src] + ((index * size) + offset)]
            inst(
                LOADW,
                register(dst),
                (index * size) + offset,
                register(src),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::LoadE(dst, src, index, offset, size) => {
            let mut result = vec![
                // register[util] <- register[index] * size
                inst(
                    FMUL,
                    utility_register,
                    register(index),
                    size,
                    AS::Register,
                    AS::Immediate,
                ),
                // register[util] <- register[src] + register[util]
                inst(
                    FADD,
                    utility_register,
                    register(src),
                    utility_register,
                    AS::Register,
                    AS::Register,
                ),
            ];

            for i in 0..EF::D {
                // register[dst] <- mem[register[util] + offset + (i * WORD_SIZE)]
                result.push(inst(
                    LOADW,
                    register(dst - ((i * WORD_SIZE) as i32)),
                    offset + F::from_canonical_usize(i * WORD_SIZE),
                    utility_register,
                    AS::Register,
                    AS::Memory,
                ))
            }
            result
        }
        AsmInstruction::LoadEI(dst, src, index, offset, size) => (0..EF::D)
            .map(|i|
                // register[dst] <- mem[register[src] + ((index * size) + offset + (i * WORD_SIZE))]
                inst(
                    LOADW,
                    register(dst - ((i * WORD_SIZE) as i32)),
                    (index * size) + offset + F::from_canonical_usize(i * WORD_SIZE),
                    register(src),
                    AS::Register,
                    AS::Memory,
                ))
            .collect(),
        AsmInstruction::StoreF(val, addr, index, offset, size) => vec![
            // register[util] <- register[index] * size
            inst(
                FMUL,
                utility_register,
                register(index),
                size,
                AS::Register,
                AS::Immediate,
            ),
            // register[util] <- register[src] + register[util]
            inst(
                FADD,
                utility_register,
                register(addr),
                utility_register,
                AS::Register,
                AS::Register,
            ),
            //  mem[register[util] + offset] <- register[val]
            inst(
                STOREW,
                register(val),
                offset,
                utility_register,
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::StoreFI(val, addr, index, offset, size) => vec![
            // mem[register[addr] + ((index * size) + offset)] <- register[val]
            inst(
                STOREW,
                register(val),
                (index * size) + offset,
                register(addr),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::StoreE(val, addr, index, offset, size) => {
            let mut result = vec![
                // register[util] <- register[index] * size
                inst(
                    FMUL,
                    utility_register,
                    register(index),
                    size,
                    AS::Register,
                    AS::Immediate,
                ),
                // register[util] <- register[src] + register[util]
                inst(
                    FADD,
                    utility_register,
                    register(addr),
                    utility_register,
                    AS::Register,
                    AS::Register,
                ),
            ];

            for i in 0..EF::D {
                // mem[register[util] + offset + (i * WORD_SIZE)] <- register[val]
                result.push(inst(
                    STOREW,
                    register(val - ((i * WORD_SIZE) as i32)),
                    offset + F::from_canonical_usize(i * WORD_SIZE),
                    utility_register,
                    AS::Register,
                    AS::Memory,
                ))
            }
            result
        }
        AsmInstruction::StoreEI(val, addr, index, offset, size) => (0..EF::D)
            .map(|i|
                // mem[register[addr] + ((index * size) + offset + (i * WORD_SIZE))] <- register[val]
                inst(
                    STOREW,
                    register(val - ((i * WORD_SIZE) as i32)),
                    (index * size) + offset + F::from_canonical_usize(i * WORD_SIZE),
                    register(addr),
                    AS::Register,
                    AS::Memory,
                ))
            .collect(),
        AsmInstruction::Jal(dst, label, offset) => {
            assert_eq!(offset, F::zero());
            vec![
                // pc <- labels[label] + offset, register[dst] <- pc
                inst(
                    JAL,
                    register(dst),
                    labels(label) - pc,
                    F::zero(),
                    AS::Register,
                    AS::Immediate,
                ),
            ]
        }
        AsmInstruction::JalR(_dst, _label, _offset) => panic!("Jalr should never be used"),
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
        AsmInstruction::BneE(label, lhs, rhs) => (0..EF::D)
            .map(|i|
            // if register[lhs + i] != register[rhs +i] for i = 0..4, pc <- labels[label]
            inst(
                BNE,
                register(lhs - ((i * WORD_SIZE) as i32)),
                register(rhs - ((i * WORD_SIZE) as i32)),
                labels(label) - (pc + F::from_canonical_usize(i)),
                AS::Register,
                AS::Register,
            ))
            .collect(),
        AsmInstruction::BneEI(label, lhs, rhs) => (0..EF::D)
            .map(|i|
            // if register[lhs + i] != rhs[i] for i = 0..4, pc <- labels[label]
            inst(
                BNE,
                register(lhs - ((i * WORD_SIZE) as i32)),
                rhs.as_base_slice()[i],
                labels(label) - (pc + F::from_canonical_usize(i)),
                AS::Register,
                AS::Register,
            ))
            .collect(),
        AsmInstruction::BeqE(label, lhs, rhs) => (0..EF::D)
            .rev()
            .map(|i|
            // if register[lhs + i] == register[rhs + i] for i = 0..4, pc <- labels[label]
            inst(
                if i == 0 { BEQ } else { BNE },
                register(lhs - ((i * WORD_SIZE) as i32)),
                register(rhs - ((i * WORD_SIZE) as i32)),
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize(EF::D - 1))
                } else {
                    F::from_canonical_usize(i + 1)
                },
                AS::Register,
                AS::Register,
            ))
            .collect(),
        AsmInstruction::BeqEI(label, lhs, rhs) => (0..EF::D)
            .rev()
            .map(|i|
            // if register[lhs + i] == rhs[i] for i = 0..4, pc <- labels[label]
            inst(
                if i == 0 { BEQ } else { BNE },
                register(lhs - ((i * WORD_SIZE) as i32)),
                rhs.as_base_slice()[i],
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize(EF::D - 1))
                } else {
                    F::from_canonical_usize(i + 1)
                },
                AS::Register,
                AS::Register,
            ))
            .collect(),
        AsmInstruction::Trap => vec![
            // pc <- -1 (causes trace generation to fail)
            inst(
                FAIL,
                F::zero(),
                F::zero(),
                F::zero(),
                AS::Immediate,
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
        AsmInstruction::HintInputVec() => vec![inst(
            HINT_INPUT,
            F::zero(),
            F::zero(),
            F::zero(),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::HintBits(src) => vec![inst(
            HINT_BITS,
            register(src),
            F::zero(),
            F::zero(),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::StoreHintWordI(val, offset, index, size) => vec![inst(
            SHINTW,
            register(val),
            (index * size) + offset,
            F::zero(),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::StoreHintWord(addr, index, offset, size) => vec![
            // register[util] <- register[index] * size
            inst(
                FMUL,
                utility_register,
                register(index),
                size,
                AS::Register,
                AS::Immediate,
            ),
            // register[util] <- register[src] + register[util]
            inst(
                FADD,
                utility_register,
                register(addr),
                utility_register,
                AS::Register,
                AS::Register,
            ),
            //  mem[register[util] + offset] <- hint_word
            inst(
                SHINTW,
                utility_register,
                offset,
                F::zero(),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::PrintV(..) | AsmInstruction::PrintF(..) | AsmInstruction::PrintE(..) => {
            if options.compile_prints {
                convert_print_instruction::<WORD_SIZE, F, EF>(instruction)
            } else {
                vec![]
            }
        }
        AsmInstruction::AddF(..)
        | AsmInstruction::SubF(..)
        | AsmInstruction::MulF(..)
        | AsmInstruction::DivF(..)
        | AsmInstruction::AddFI(..)
        | AsmInstruction::SubFI(..)
        | AsmInstruction::MulFI(..)
        | AsmInstruction::DivFI(..)
        | AsmInstruction::SubFIN(..)
        | AsmInstruction::DivFIN(..) => {
            if options.field_arithmetic_enabled {
                convert_base_arithmetic_instruction(instruction, utility_register)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::AddE(..)
        | AsmInstruction::AddEI(..)
        | AsmInstruction::SubE(..)
        | AsmInstruction::SubEI(..)
        | AsmInstruction::SubEIN(..)
        | AsmInstruction::MulE(..)
        | AsmInstruction::MulEI(..)
        | AsmInstruction::DivE(..)
        | AsmInstruction::DivEI(..)
        | AsmInstruction::DivEIN(..) => {
            let fe_utility_registers = from_fn(|i| utility_registers[i]);
            if options.field_extension_enabled {
                convert_field_extension::<WORD_SIZE, F, EF>(instruction, fe_utility_registers)
            } else if options.field_arithmetic_enabled {
                convert_field_extension_with_base::<WORD_SIZE, F, EF>(
                    instruction,
                    fe_utility_registers,
                )
            } else {
                panic!(
                    "Unsupported instruction {:?}, field extension arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::Poseidon2Compress(src1, src2, dst) => vec![inst(
            COMP_POS2,
            register(src1),
            register(src2),
            register(dst),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::Poseidon2Permute(src, dst) => vec![
            inst(
                FADD,
                utility_register,
                register(src),
                F::from_canonical_usize(POSEIDON2_WIDTH / 2),
                AS::Register,
                AS::Immediate,
            ),
            inst(
                PERM_POS2,
                register(src),
                utility_register,
                register(dst),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::CycleTracker(_) => vec![],
        _ => panic!("Unsupported instruction {:?}", instruction),
    }
}

pub fn convert_program<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    program: AssemblyCode<F, EF>,
    options: CompilerOptions,
) -> Vec<Instruction<F>> {
    // register[0] <- 0
    let init_register_0 = inst(
        STOREW,
        F::zero(),
        F::zero(),
        register(0),
        AS::Immediate,
        AS::Register,
    );

    let mut block_start = vec![];
    let mut pc = 1;
    for block in program.blocks.iter() {
        block_start.push(pc);
        for instruction in block.0.iter() {
            let instructions = convert_instruction::<WORD_SIZE, F, EF>(
                instruction.clone(),
                F::from_canonical_usize(pc),
                |label| label,
                options,
            );
            pc += instructions.len();
        }
    }

    let mut result = vec![init_register_0];
    for block in program.blocks.iter() {
        for instruction in block.0.iter() {
            let labels =
                |label: F| F::from_canonical_usize(block_start[label.as_canonical_u64() as usize]);
            result.extend(convert_instruction::<WORD_SIZE, F, EF>(
                instruction.clone(),
                F::from_canonical_usize(result.len()),
                labels,
                options,
            ));
        }
    }

    result
}
