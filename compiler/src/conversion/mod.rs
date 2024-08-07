use p3_field::{ExtensionField, PrimeField64};
use stark_vm::{
    cpu::{
        trace::Instruction,
        OpCode::{self, *},
    },
    program::{DebugInfo, Program},
};

use crate::asm::{AsmInstruction, AssemblyCode};

#[derive(Clone, Copy, Debug)]
pub struct CompilerOptions {
    pub compile_prints: bool,
    pub enable_cycle_tracker: bool,
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            compile_prints: true,
            enable_cycle_tracker: false,
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
        }
    }
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
        debug: String::new(),
    }
}

fn dbg<F: PrimeField64>(opcode: OpCode, debug: String) -> Instruction<F> {
    Instruction {
        opcode,
        op_a: F::zero(),
        op_b: F::zero(),
        op_c: F::zero(),
        d: F::zero(),
        e: F::zero(),
        debug,
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

fn register<F: PrimeField64>(value: i32) -> F {
    assert!(value <= 0);
    F::from_canonical_usize(-value as usize)
}

fn convert_base_arithmetic_instruction<F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
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
        _ => panic!(
            "Illegal argument to convert_field_arithmetic_instruction: {:?}",
            instruction
        ),
    }
}

pub fn convert_field_extension<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
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
    debug_info: Option<DebugInfo>,
    pc: F,
    labels: impl Fn(F) -> F,
    options: CompilerOptions,
) -> Program<F> {
    let instructions = match instruction {
        AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
        AsmInstruction::LoadFI(dst, src, offset) => vec![
            // register[dst] <- mem[register[src] + offset]
            inst(
                LOADW,
                register(dst),
                offset,
                register(src),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::StoreFI(val, addr, offset) => vec![
            // mem[register[addr] + offset] <- register[val]
            inst(
                STOREW,
                register(val),
                offset,
                register(addr),
                AS::Register,
                AS::Memory,
            ),
        ],
        AsmInstruction::Jump(dst, label) => {
            vec![
                // pc <- labels[label], register[dst] <- pc
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
                AS::Immediate,
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
                AS::Immediate,
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
        AsmInstruction::StoreHintWordI(val, offset) => vec![inst(
            SHINTW,
            register(val),
            offset,
            F::zero(),
            AS::Register,
            AS::Memory,
        )],
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
        | AsmInstruction::DivFI(..) => {
            if options.field_arithmetic_enabled {
                convert_base_arithmetic_instruction(instruction)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::AddE(..)
        | AsmInstruction::SubE(..)
        | AsmInstruction::MulE(..)
        | AsmInstruction::InvE(..) => {
            if options.field_extension_enabled {
                convert_field_extension::<WORD_SIZE, F, EF>(instruction)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field extension arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::Poseidon2Compress(dst, src1, src2) => vec![inst(
            COMP_POS2,
            register(dst),
            register(src1),
            register(src2),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::Poseidon2Permute(dst, src) => vec![inst(
            PERM_POS2,
            register(dst),
            register(src),
            F::zero(),
            AS::Register,
            AS::Memory,
        )],
        AsmInstruction::CycleTrackerStart(name) => {
            if options.enable_cycle_tracker {
                vec![dbg(CT_START, name)]
            } else {
                vec![]
            }
        }
        AsmInstruction::CycleTrackerEnd(name) => {
            if options.enable_cycle_tracker {
                vec![dbg(CT_END, name)]
            } else {
                vec![]
            }
        }
        AsmInstruction::Publish(val, index) => vec![inst(
            PUBLISH,
            register(index),
            register(val),
            F::zero(),
            AS::Register,
            AS::Register,
        )],
        _ => panic!("Unsupported instruction {:?}", instruction),
    };

    let debug_infos = vec![debug_info; instructions.len()];
    Program {
        instructions,
        debug_infos,
    }
}

pub fn convert_program<const WORD_SIZE: usize, F: PrimeField64, EF: ExtensionField<F>>(
    program: AssemblyCode<F, EF>,
    options: CompilerOptions,
) -> Program<F> {
    // register[0] <- 0
    let init_register_0 = inst(
        STOREW,
        F::zero(),
        F::zero(),
        register(0),
        AS::Immediate,
        AS::Register,
    );
    let init_debug_info = None;

    let mut block_start = vec![];
    let mut pc = 1;
    for block in program.blocks.iter() {
        block_start.push(pc);

        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            let instructions = convert_instruction::<WORD_SIZE, F, EF>(
                instruction.clone(),
                debug_info.clone(),
                F::from_canonical_usize(pc),
                |label| label,
                options,
            );
            pc += instructions.len();
        }
    }

    let mut instructions = vec![init_register_0];
    let mut debug_infos = vec![init_debug_info];
    for block in program.blocks.iter() {
        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            let labels =
                |label: F| F::from_canonical_usize(block_start[label.as_canonical_u64() as usize]);
            let result = convert_instruction::<WORD_SIZE, F, EF>(
                instruction.clone(),
                debug_info.clone(),
                F::from_canonical_usize(instructions.len()),
                labels,
                options,
            );
            instructions.extend(result.instructions);
            debug_infos.extend(result.debug_infos);
        }
    }

    Program {
        instructions,
        debug_infos,
    }
}
