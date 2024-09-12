use p3_field::{ExtensionField, PrimeField32, PrimeField64};
use stark_vm::{
    arch::instructions::Opcode::{self, *},
    cpu::trace::Instruction,
    program::{DebugInfo, Program},
};

use crate::asm::{AsmInstruction, AssemblyCode};

#[derive(Clone, Copy, Debug)]
pub struct CompilerOptions {
    pub compile_prints: bool,
    pub enable_cycle_tracker: bool,
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub field_less_than_enabled: bool,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            compile_prints: true,
            enable_cycle_tracker: false,
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            field_less_than_enabled: false,
        }
    }
}

fn inst<F: PrimeField64>(
    opcode: Opcode,
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
        op_f: F::zero(),
        op_g: F::zero(),
        debug: String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_med<F: PrimeField64>(
    opcode: Opcode,
    op_a: F,
    op_b: F,
    op_c: F,
    d: AS,
    e: AS,
    f: AS,
) -> Instruction<F> {
    Instruction {
        opcode,
        op_a,
        op_b,
        op_c,
        d: d.to_field(),
        e: e.to_field(),
        op_f: f.to_field(),
        op_g: F::zero(),
        debug: String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_large<F: PrimeField64>(
    opcode: Opcode,
    op_a: F,
    op_b: F,
    op_c: F,
    d: AS,
    e: AS,
    op_f: F,
    op_g: F,
) -> Instruction<F> {
    Instruction {
        opcode,
        op_a,
        op_b,
        op_c,
        d: d.to_field(),
        e: e.to_field(),
        op_f,
        op_g,
        debug: String::new(),
    }
}

fn dbg<F: PrimeField64>(opcode: Opcode, debug: String) -> Instruction<F> {
    Instruction {
        opcode,
        op_a: F::zero(),
        op_b: F::zero(),
        op_c: F::zero(),
        d: F::zero(),
        e: F::zero(),
        op_f: F::zero(),
        op_g: F::zero(),
        debug,
    }
}

#[derive(Clone, Copy)]
enum AS {
    Immediate,
    #[allow(dead_code)]
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

fn i32_f<F: PrimeField32>(x: i32) -> F {
    let modulus = F::ORDER_U32;
    assert!(x < modulus as i32 && x >= -(modulus as i32));
    if x < 0 {
        -F::from_canonical_u32((-x) as u32)
    } else {
        F::from_canonical_u32(x as u32)
    }
}

fn convert_comparison_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::LessThanF(dst, lhs, rhs) => vec![inst(
            F_LESS_THAN,
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::LessThanFI(dst, lhs, rhs) => vec![inst(
            F_LESS_THAN,
            i32_f(dst),
            i32_f(lhs),
            rhs,
            AS::Memory,
            AS::Immediate,
        )],
        AsmInstruction::EqU256(a, b, c) => vec![inst_large(
            EQ256,
            i32_f(a),
            i32_f(b),
            i32_f(c),
            AS::Memory,
            AS::Memory,
            AS::Memory.to_field(),
            AS::Memory.to_field(),
        )],
        _ => panic!(
            "Illegal argument to convert_comparison_instruction: {:?}",
            instruction
        ),
    }
}

fn convert_base_arithmetic_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::AddF(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] + mem[rhs]
            inst_med(
                FADD,
                i32_f(dst),
                i32_f(lhs),
                i32_f(rhs),
                AS::Memory,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::AddFI(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] + rhs
            inst_med(
                FADD,
                i32_f(dst),
                i32_f(lhs),
                rhs,
                AS::Memory,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::SubF(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] - mem[rhs]
            inst_med(
                FSUB,
                i32_f(dst),
                i32_f(lhs),
                i32_f(rhs),
                AS::Memory,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::SubFI(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] - rhs
            inst_med(
                FSUB,
                i32_f(dst),
                i32_f(lhs),
                rhs,
                AS::Memory,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::SubFIN(dst, lhs, rhs) => vec![
            // mem[dst] <- lhs - mem[rhs]
            inst_med(
                FSUB,
                i32_f(dst),
                lhs,
                i32_f(rhs),
                AS::Memory,
                AS::Immediate,
                AS::Memory,
            ),
        ],
        AsmInstruction::MulF(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] * mem[rhs]
            inst_med(
                FMUL,
                i32_f(dst),
                i32_f(lhs),
                i32_f(rhs),
                AS::Memory,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::MulFI(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] * rhs
            inst_med(
                FMUL,
                i32_f(dst),
                i32_f(lhs),
                rhs,
                AS::Memory,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::DivF(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] / mem[rhs]
            inst_med(
                FDIV,
                i32_f(dst),
                i32_f(lhs),
                i32_f(rhs),
                AS::Memory,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::DivFI(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] / rhs
            inst_med(
                FDIV,
                i32_f(dst),
                i32_f(lhs),
                rhs,
                AS::Memory,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::DivFIN(dst, lhs, rhs) => vec![
            // mem[dst] <- lhs / mem[rhs]
            inst_med(
                FDIV,
                i32_f(dst),
                lhs,
                i32_f(rhs),
                AS::Memory,
                AS::Immediate,
                AS::Memory,
            ),
        ],
        _ => panic!(
            "Illegal argument to convert_field_arithmetic_instruction: {:?}",
            instruction
        ),
    }
}

pub fn convert_field_extension<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::AddE(dst, lhs, rhs) => vec![inst(
            FE4ADD,
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::SubE(dst, lhs, rhs) => vec![inst(
            FE4SUB,
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::MulE(dst, lhs, rhs) => vec![inst(
            BBE4MUL,
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::DivE(dst, lhs, rhs) => vec![inst(
            BBE4DIV,
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        _ => panic!(
            "Illegal argument to convert_field_extension: {:?}",
            instruction
        ),
    }
}

fn convert_print_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
) -> Vec<Instruction<F>> {
    let word_size_i32 = 1;

    match instruction {
        AsmInstruction::PrintV(src) => vec![inst(
            PRINTF,
            i32_f(src),
            F::zero(),
            F::zero(),
            AS::Memory,
            AS::Immediate,
        )],
        AsmInstruction::PrintF(src) => vec![inst(
            PRINTF,
            i32_f(src),
            F::zero(),
            F::zero(),
            AS::Memory,
            AS::Immediate,
        )],
        AsmInstruction::PrintE(src) => vec![
            inst(
                PRINTF,
                i32_f(src),
                F::zero(),
                F::zero(),
                AS::Memory,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                i32_f(src + word_size_i32),
                F::zero(),
                F::zero(),
                AS::Memory,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                i32_f(src + 2 * word_size_i32),
                F::zero(),
                F::zero(),
                AS::Memory,
                AS::Immediate,
            ),
            inst(
                PRINTF,
                i32_f(src + 3 * word_size_i32),
                F::zero(),
                F::zero(),
                AS::Memory,
                AS::Immediate,
            ),
        ],
        _ => panic!(
            "Illegal argument to convert_print_instruction: {:?}",
            instruction
        ),
    }
}

fn convert_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    debug_info: Option<DebugInfo>,
    pc: F,
    labels: impl Fn(F) -> F,
    options: CompilerOptions,
) -> Program<F> {
    let instructions = match instruction {
        AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
        AsmInstruction::LoadF(dst, src, index, size, offset) => vec![
            // mem[dst] <- mem[mem[src] + mem[index] * size + offset]
            inst_large(
                LOADW2,
                i32_f(dst),
                offset,
                i32_f(src),
                AS::Memory,
                AS::Memory,
                i32_f(index),
                size,
            ),
        ],
        AsmInstruction::LoadFI(dst, src, index, size, offset) => vec![
            // mem[dst] <- mem[mem[src] + index * size + offset]
            inst(
                LOADW,
                i32_f(dst),
                index * size + offset,
                i32_f(src),
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::StoreF(val, addr, index, size, offset) => vec![
            // mem[mem[addr] + mem[index] * size + offset] <- mem[val]
            inst_large(
                STOREW2,
                i32_f(val),
                offset,
                i32_f(addr),
                AS::Memory,
                AS::Memory,
                i32_f(index),
                size,
            ),
        ],
        AsmInstruction::StoreFI(val, addr, index, size, offset) => vec![
            // mem[mem[addr] + index * size + offset] <- mem[val]
            inst(
                STOREW,
                i32_f(val),
                index * size + offset,
                i32_f(addr),
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::Jump(dst, label) => {
            vec![
                // pc <- labels[label], mem[dst] <- pc
                inst(
                    JAL,
                    i32_f(dst),
                    labels(label) - pc,
                    F::zero(),
                    AS::Memory,
                    AS::Immediate,
                ),
            ]
        }
        AsmInstruction::Bne(label, lhs, rhs) => vec![
            // if mem[lhs] != mem[rhs], pc <- labels[label]
            inst(
                BNE,
                i32_f(lhs),
                i32_f(rhs),
                labels(label) - pc,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::BneI(label, lhs, rhs) => vec![
            // if mem[lhs] != rhs, pc <- labels[label]
            inst(
                BNE,
                i32_f(lhs),
                rhs,
                labels(label) - pc,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::Beq(label, lhs, rhs) => vec![
            // if mem[lhs] == mem[rhs], pc <- labels[label]
            inst(
                BEQ,
                i32_f(lhs),
                i32_f(rhs),
                labels(label) - pc,
                AS::Memory,
                AS::Memory,
            ),
        ],
        AsmInstruction::BeqI(label, lhs, rhs) => vec![
            // if mem[lhs] == rhs, pc <- labels[label]
            inst(
                BEQ,
                i32_f(lhs),
                rhs,
                labels(label) - pc,
                AS::Memory,
                AS::Immediate,
            ),
        ],
        AsmInstruction::BneE(label, lhs, rhs) => (0..EF::D)
            .map(|i|
            // if mem[lhs + i] != mem[rhs +i] for i = 0..4, pc <- labels[label]
            inst(
                BNE,
                i32_f(lhs + (i as i32)),
                i32_f(rhs + (i as i32)),
                labels(label) - (pc + F::from_canonical_usize(i)),
                AS::Memory,
                AS::Memory,
            ))
            .collect(),
        AsmInstruction::BneEI(label, lhs, rhs) => (0..EF::D)
            .map(|i|
            // if mem[lhs + i] != rhs[i] for i = 0..4, pc <- labels[label]
            inst(
                BNE,
                i32_f(lhs + (i as i32)),
                rhs.as_base_slice()[i],
                labels(label) - (pc + F::from_canonical_usize(i)),
                AS::Memory,
                AS::Immediate,
            ))
            .collect(),
        AsmInstruction::BeqE(label, lhs, rhs) => (0..EF::D)
            .rev()
            .map(|i|
            // if mem[lhs + i] == mem[rhs + i] for i = 0..4, pc <- labels[label]
            inst(
                if i == 0 { BEQ } else { BNE },
                i32_f(lhs + (i as i32)),
                i32_f(rhs + (i as i32)),
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize(EF::D - 1))
                } else {
                    F::from_canonical_usize(i + 1)
                },
                AS::Memory,
                AS::Memory,
            ))
            .collect(),
        AsmInstruction::BeqEI(label, lhs, rhs) => (0..EF::D)
            .rev()
            .map(|i|
            // if mem[lhs + i] == rhs[i] for i = 0..4, pc <- labels[label]
            inst(
                if i == 0 { BEQ } else { BNE },
                i32_f(lhs + (i as i32)),
                rhs.as_base_slice()[i],
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize(EF::D - 1))
                } else {
                    F::from_canonical_usize(i + 1)
                },
                AS::Memory,
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
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::HintBits(src, len) => vec![inst(
            HINT_BITS,
            i32_f(src),
            F::zero(),
            F::from_canonical_u32(len),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::HintBytes(src, len) => vec![inst(
            HINT_BYTES,
            i32_f(src),
            F::zero(),
            F::from_canonical_u32(len),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::StoreHintWordI(val, offset) => vec![inst(
            SHINTW,
            i32_f(val),
            offset,
            F::zero(),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::PrintV(..) | AsmInstruction::PrintF(..) | AsmInstruction::PrintE(..) => {
            if options.compile_prints {
                convert_print_instruction(instruction)
            } else {
                vec![]
            }
        }
        AsmInstruction::ImmF(dst, val) => vec![inst(
            STOREW,
            val,
            F::zero(),
            i32_f(dst),
            AS::Immediate,
            AS::Memory,
        )],
        AsmInstruction::CopyF(dst, src) => vec![inst(
            LOADW,
            i32_f(dst),
            F::zero(),
            i32_f(src),
            AS::Memory,
            AS::Immediate,
        )],
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
                convert_base_arithmetic_instruction(instruction)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::LessThanF(..) | AsmInstruction::LessThanFI(..) => {
            if options.field_less_than_enabled {
                convert_comparison_instruction(instruction)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field less than is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::EqU256(..) => convert_comparison_instruction(instruction),
        AsmInstruction::AddE(..)
        | AsmInstruction::SubE(..)
        | AsmInstruction::MulE(..)
        | AsmInstruction::DivE(..) => {
            if options.field_extension_enabled {
                convert_field_extension(instruction)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field extension arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::Poseidon2Compress(dst, src1, src2) => vec![inst(
            COMP_POS2,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Poseidon2Permute(dst, src) => vec![inst(
            PERM_POS2,
            i32_f(dst),
            i32_f(src),
            F::zero(),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::AddSecp256k1Coord(dst, src1, src2) => vec![inst(
            SECP256K1_COORD_ADD,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::SubSecp256k1Coord(dst, src1, src2) => vec![inst(
            SECP256K1_COORD_SUB,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::MulSecp256k1Coord(dst, src1, src2) => vec![inst(
            SECP256K1_COORD_MUL,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::DivSecp256k1Coord(dst, src1, src2) => vec![inst(
            SECP256K1_COORD_DIV,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::AddSecp256k1Scalar(dst, src1, src2) => vec![inst(
            SECP256K1_SCALAR_ADD,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::SubSecp256k1Scalar(dst, src1, src2) => vec![inst(
            SECP256K1_SCALAR_SUB,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::MulSecp256k1Scalar(dst, src1, src2) => vec![inst(
            SECP256K1_SCALAR_MUL,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::DivSecp256k1Scalar(dst, src1, src2) => vec![inst(
            SECP256K1_SCALAR_DIV,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::AddU256(dst, src1, src2) => vec![inst_large(
            ADD256,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
            AS::Memory.to_field(),
            AS::Memory.to_field(),
        )],
        AsmInstruction::SubU256(dst, src1, src2) => vec![inst_large(
            SUB256,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
            AS::Memory.to_field(),
            AS::Memory.to_field(),
        )],
        AsmInstruction::LessThanU256(dst, src1, src2) => vec![inst_large(
            LT256,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
            AS::Memory.to_field(),
            AS::Memory.to_field(),
        )],
        AsmInstruction::EqualToU256(dst, src1, src2) => vec![inst_large(
            EQ256,
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
            AS::Memory.to_field(),
            AS::Memory.to_field(),
        )],
        AsmInstruction::Keccak256(dst, src, len) => vec![inst_med(
            KECCAK256,
            i32_f(dst),
            i32_f(src),
            i32_f(len),
            AS::Memory,
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Keccak256FixLen(_dst, _src, _len) => {
            todo!("len as immediate needs to be handled");
            // inst_med(
            //     KECCAK256,
            //     i32_f(dst),
            //     i32_f(src),
            //     i32_f(len),
            //     AS::Memory,
            //     AS::Memory,
            //     AS::Immediate,
            // )
        }
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
            i32_f(index),
            i32_f(val),
            F::zero(),
            AS::Memory,
            AS::Memory,
        )],
    };

    let debug_infos = vec![debug_info; instructions.len()];
    Program {
        instructions,
        debug_infos,
    }
}

pub fn convert_program<F: PrimeField32, EF: ExtensionField<F>>(
    program: AssemblyCode<F, EF>,
    options: CompilerOptions,
) -> Program<F> {
    // mem[0] <- 0
    let init_register_0 = inst(
        STOREW,
        F::zero(),
        F::zero(),
        i32_f(0),
        AS::Immediate,
        AS::Memory,
    );
    let init_debug_info = None;

    let mut block_start = vec![];
    let mut pc = 1;
    for block in program.blocks.iter() {
        block_start.push(pc);

        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            let instructions = convert_instruction::<F, EF>(
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
            let result = convert_instruction(
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
