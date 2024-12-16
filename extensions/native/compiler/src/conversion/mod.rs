use openvm_circuit::arch::instructions::program::Program;
use openvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::{DEFAULT_MAX_NUM_PUBLIC_VALUES, DEFAULT_PC_STEP},
    PhantomDiscriminant, Poseidon2Opcode, PublishOpcode, SysPhantom, SystemOpcode, UsizeOpcode,
    VmOpcode,
};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::{ExtensionField, PrimeField32, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::{
    asm::{AsmInstruction, AssemblyCode},
    FieldArithmeticOpcode, FieldExtensionOpcode, FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode, NativeLoadStoreOpcode, NativePhantom,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CompilerOptions {
    // The compiler will ensure that the heap pointer is aligned to be a multiple of `word_size`.
    pub word_size: usize,
    pub compile_prints: bool,
    pub enable_cycle_tracker: bool,
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            word_size: 8,
            compile_prints: true,
            enable_cycle_tracker: false,
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
        }
    }
}

impl CompilerOptions {
    pub fn opcode_with_offset<Opcode: UsizeOpcode>(&self, opcode: Opcode) -> VmOpcode {
        let offset = Opcode::default_offset();
        VmOpcode::from_usize(offset + opcode.as_usize())
    }
    pub fn with_cycle_tracker(mut self) -> Self {
        self.enable_cycle_tracker = true;
        self
    }
}

fn inst<F: PrimeField64>(opcode: VmOpcode, a: F, b: F, c: F, d: AS, e: AS) -> Instruction<F> {
    Instruction {
        opcode,
        a,
        b,
        c,
        d: d.to_field(),
        e: e.to_field(),
        f: F::ZERO,
        g: F::ZERO,
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_med<F: PrimeField64>(
    opcode: VmOpcode,
    a: F,
    b: F,
    c: F,
    d: AS,
    e: AS,
    f: AS,
) -> Instruction<F> {
    Instruction {
        opcode,
        a,
        b,
        c,
        d: d.to_field(),
        e: e.to_field(),
        f: f.to_field(),
        g: F::ZERO,
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_large<F: PrimeField64>(
    opcode: VmOpcode,
    a: F,
    b: F,
    c: F,
    d: AS,
    e: AS,
    f: F,
    g: F,
) -> Instruction<F> {
    Instruction {
        opcode,
        a,
        b,
        c,
        d: d.to_field(),
        e: e.to_field(),
        f,
        g,
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
    fn to_field<F: PrimeField64>(self) -> F {
        match self {
            AS::Immediate => F::ZERO,
            AS::Register => F::ONE,
            AS::Memory => F::TWO,
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

fn convert_base_arithmetic_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    options: &CompilerOptions,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::AddF(dst, lhs, rhs) => vec![
            // mem[dst] <- mem[lhs] + mem[rhs]
            inst_med(
                options.opcode_with_offset(FieldArithmeticOpcode::ADD),
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
                options.opcode_with_offset(FieldArithmeticOpcode::ADD),
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
                options.opcode_with_offset(FieldArithmeticOpcode::SUB),
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
                options.opcode_with_offset(FieldArithmeticOpcode::SUB),
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
                options.opcode_with_offset(FieldArithmeticOpcode::SUB),
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
                options.opcode_with_offset(FieldArithmeticOpcode::MUL),
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
                options.opcode_with_offset(FieldArithmeticOpcode::MUL),
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
                options.opcode_with_offset(FieldArithmeticOpcode::DIV),
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
                options.opcode_with_offset(FieldArithmeticOpcode::DIV),
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
                options.opcode_with_offset(FieldArithmeticOpcode::DIV),
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
    options: &CompilerOptions,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::AddE(dst, lhs, rhs) => vec![inst(
            options.opcode_with_offset(FieldExtensionOpcode::FE4ADD),
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::SubE(dst, lhs, rhs) => vec![inst(
            options.opcode_with_offset(FieldExtensionOpcode::FE4SUB),
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::MulE(dst, lhs, rhs) => vec![inst(
            options.opcode_with_offset(FieldExtensionOpcode::BBE4MUL),
            i32_f(dst),
            i32_f(lhs),
            i32_f(rhs),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::DivE(dst, lhs, rhs) => vec![inst(
            options.opcode_with_offset(FieldExtensionOpcode::BBE4DIV),
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
    _options: &CompilerOptions,
) -> Vec<Instruction<F>> {
    let word_size_i32 = 1;

    match instruction {
        AsmInstruction::PrintV(src) => vec![Instruction::phantom(
            PhantomDiscriminant(NativePhantom::Print as u16),
            i32_f(src),
            F::ZERO,
            2,
        )],
        AsmInstruction::PrintF(src) => vec![Instruction::phantom(
            PhantomDiscriminant(NativePhantom::Print as u16),
            i32_f(src),
            F::ZERO,
            2,
        )],
        AsmInstruction::PrintE(src) => (0..EF::D as i32)
            .map(|i| {
                Instruction::phantom(
                    PhantomDiscriminant(NativePhantom::Print as u16),
                    i32_f(src + i * word_size_i32),
                    F::ZERO,
                    2,
                )
            })
            .collect(),
        _ => panic!(
            "Illegal argument to convert_print_instruction: {:?}",
            instruction
        ),
    }
}

/// Warning: for extension field branch instructions, the `pc, labels` **must** be using `DEFAULT_PC_STEP`.
fn convert_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    debug_info: Option<DebugInfo>,
    pc: F,
    labels: impl Fn(F) -> F,
    options: &CompilerOptions,
) -> Program<F> {
    let instructions = match instruction {
        AsmInstruction::Break(_) => panic!("Unresolved break instruction"),
        AsmInstruction::LoadF(dst, src, index, size, offset) => vec![
            // mem[dst] <- mem[mem[src] + mem[index] * size + offset]
            inst_large(
                options.opcode_with_offset(NativeLoadStoreOpcode::LOADW2),
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
                options.opcode_with_offset(NativeLoadStoreOpcode::LOADW),
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
                options.opcode_with_offset(NativeLoadStoreOpcode::STOREW2),
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
                options.opcode_with_offset(NativeLoadStoreOpcode::STOREW),
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
                    options.opcode_with_offset(NativeJalOpcode::JAL),
                    i32_f(dst),
                    labels(label) - pc,
                    F::ZERO,
                    AS::Memory,
                    AS::Immediate,
                ),
            ]
        }
        AsmInstruction::Bne(label, lhs, rhs) => vec![
            // if mem[lhs] != mem[rhs], pc <- labels[label]
            inst(
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)),
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
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)),
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
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BEQ)),
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
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BEQ)),
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
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)),
                i32_f(lhs + (i as i32)),
                i32_f(rhs + (i as i32)),
                labels(label) - (pc + F::from_canonical_usize(i * DEFAULT_PC_STEP as usize)),
                AS::Memory,
                AS::Memory,
            ))
            .collect(),
        AsmInstruction::BneEI(label, lhs, rhs) => (0..EF::D)
            .map(|i|
            // if mem[lhs + i] != rhs[i] for i = 0..4, pc <- labels[label]
            inst(
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)),
                i32_f(lhs + (i as i32)),
                rhs.as_base_slice()[i],
                labels(label) - (pc + F::from_canonical_usize(i * DEFAULT_PC_STEP as usize)),
                AS::Memory,
                AS::Immediate,
            ))
            .collect(),
        AsmInstruction::BeqE(label, lhs, rhs) => (0..EF::D)
            .rev()
            .map(|i|
            // if mem[lhs + i] == mem[rhs + i] for i = 0..4, pc <- labels[label]
            inst(
                if i == 0 { options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BEQ)) } else { options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)) },
                i32_f(lhs + (i as i32)),
                i32_f(rhs + (i as i32)),
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize((EF::D - 1) * DEFAULT_PC_STEP as usize))
                } else {
                    F::from_canonical_usize((i + 1) * DEFAULT_PC_STEP as usize)
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
                if i == 0 { options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BEQ)) } else { options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)) },
                i32_f(lhs + (i as i32)),
                rhs.as_base_slice()[i],
                if i == 0 {
                    labels(label) - (pc + F::from_canonical_usize((EF::D - 1) * DEFAULT_PC_STEP as usize))
                } else {
                    F::from_canonical_usize((i + 1) * DEFAULT_PC_STEP as usize)
                },
                AS::Memory,
                AS::Immediate,
            ))
            .collect(),
        AsmInstruction::Trap => vec![
            Instruction::phantom(PhantomDiscriminant(SysPhantom::DebugPanic as u16), F::ZERO, F::ZERO, 0),
        ],
        AsmInstruction::Halt => vec![
            // terminate
            inst(
                options.opcode_with_offset(SystemOpcode::TERMINATE),
                F::ZERO,
                F::ZERO,
                F::ZERO,
                AS::Immediate,
                AS::Immediate,
            ),
        ],
        AsmInstruction::HintInputVec() => vec![
            Instruction::phantom(PhantomDiscriminant(NativePhantom::HintInput as u16), F::ZERO, F::ZERO, 0)
        ],
        AsmInstruction::HintBits(src, len) => vec![
            Instruction::phantom(PhantomDiscriminant(NativePhantom::HintBits as u16), i32_f(src), F::from_canonical_u32(len), AS::Memory as u16)
        ],
        AsmInstruction::StoreHintWordI(val, offset) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::SHINTW),
            F::ZERO,
            offset,
            i32_f(val),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::PrintV(..) | AsmInstruction::PrintF(..) | AsmInstruction::PrintE(..) => {
            if options.compile_prints {
                convert_print_instruction(instruction, options)
            } else {
                vec![]
            }
        }
        AsmInstruction::ImmF(dst, val) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::STOREW),
            val,
            F::ZERO,
            i32_f(dst),
            AS::Immediate,
            AS::Memory,
        )],
        AsmInstruction::CopyF(dst, src) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::LOADW),
            i32_f(dst),
            F::ZERO,
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
                convert_base_arithmetic_instruction(instruction, options)
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
        | AsmInstruction::DivE(..) => {
            if options.field_extension_enabled {
                convert_field_extension(instruction, options)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field extension arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::Poseidon2Compress(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(Poseidon2Opcode::COMP_POS2),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Poseidon2Permute(dst, src) => vec![inst(
            options.opcode_with_offset(Poseidon2Opcode::PERM_POS2),
            i32_f(dst),
            i32_f(src),
            F::ZERO,
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::CycleTrackerStart() => {
            if options.enable_cycle_tracker {
                vec![Instruction::debug(PhantomDiscriminant(SysPhantom::CtStart as u16))]
            } else {
                vec![]
            }
        }
        AsmInstruction::CycleTrackerEnd() => {
            if options.enable_cycle_tracker {
                vec![Instruction::debug(PhantomDiscriminant(SysPhantom::CtEnd as u16))]
            } else {
                vec![]
            }
        }
        AsmInstruction::Publish(val, index) => vec![inst_med(
            options.opcode_with_offset(PublishOpcode::PUBLISH),
            F::ZERO,
            i32_f(val),
            i32_f(index),
            AS::Immediate,
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::FriReducedOpening(a, b, res, len, alpha, alpha_pow) => vec![Instruction {
            opcode: options.opcode_with_offset(FriOpcode::FRI_REDUCED_OPENING),
            a: i32_f(a),
            b: i32_f(b),
            c: i32_f(res),
            d: AS::Memory.to_field(),
            e: i32_f(len),
            f: i32_f(alpha),
            g: i32_f(alpha_pow),
        }],
    };

    let debug_infos = vec![debug_info; instructions.len()];

    Program::from_instructions_and_debug_infos(&instructions, &debug_infos)
}

pub fn convert_program<F: PrimeField32, EF: ExtensionField<F>>(
    program: AssemblyCode<F, EF>,
    options: CompilerOptions,
) -> Program<F> {
    // mem[0] <- 0
    let init_register_0 = inst(
        options.opcode_with_offset(NativeLoadStoreOpcode::STOREW),
        F::ZERO,
        F::ZERO,
        i32_f(0),
        AS::Immediate,
        AS::Memory,
    );
    let init_debug_info = None;

    let mut block_start = vec![];
    let mut pc_idx = 1;
    for block in program.blocks.iter() {
        block_start.push(pc_idx * DEFAULT_PC_STEP);

        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            // This is used to just to get the number of instructions in the block
            let instructions = convert_instruction::<F, EF>(
                instruction.clone(),
                debug_info.clone(),
                F::from_canonical_u32(pc_idx * DEFAULT_PC_STEP),
                |label| label,
                &options,
            );
            pc_idx += instructions.len() as u32;
        }
    }

    let mut result = Program::new_empty(DEFAULT_PC_STEP, 0, DEFAULT_MAX_NUM_PUBLIC_VALUES);
    result.push_instruction_and_debug_info(init_register_0, init_debug_info);
    for block in program.blocks.iter() {
        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            let cur_size = result.len() as u32;
            let cur_pc = cur_size * DEFAULT_PC_STEP;

            let labels =
                |label: F| F::from_canonical_u32(block_start[label.as_canonical_u64() as usize]);
            let local_result = convert_instruction(
                instruction.clone(),
                debug_info.clone(),
                F::from_canonical_u32(cur_pc),
                labels,
                &options,
            );

            result.append(local_result);
        }
    }

    result
}
