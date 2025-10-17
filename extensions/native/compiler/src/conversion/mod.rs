use openvm_circuit::arch::instructions::program::Program;
use openvm_instructions::{
    instruction::{DebugInfo, Instruction},
    program::DEFAULT_PC_STEP,
    LocalOpcode, PhantomDiscriminant, PublishOpcode, SysPhantom, SystemOpcode, VmOpcode,
};
use openvm_rv32im_transpiler::BranchEqualOpcode;
use openvm_stark_backend::p3_field::{ExtensionField, PrimeField32, PrimeField64};
use serde::{Deserialize, Serialize};

use crate::{
    asm::{AsmInstruction, AssemblyCode},
    FieldArithmeticOpcode, FieldExtensionOpcode, FriOpcode, NativeBranchEqualOpcode,
    NativeJalOpcode, NativeLoadStore4Opcode, NativeLoadStoreOpcode, NativePhantom,
    NativeRangeCheckOpcode, Poseidon2Opcode, VerifyBatchOpcode,
};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct CompilerOptions {
    // The compiler will ensure that the heap pointer is aligned to be a multiple of `word_size`.
    pub word_size: usize,
    pub enable_cycle_tracker: bool,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            word_size: 8,
            enable_cycle_tracker: false,
        }
    }
}

impl CompilerOptions {
    pub fn opcode_with_offset<Opcode: LocalOpcode>(&self, opcode: Opcode) -> VmOpcode {
        let offset = Opcode::CLASS_OFFSET;
        VmOpcode::from_usize(offset + opcode.local_usize())
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

#[derive(Clone, Copy)]
#[repr(u8)]
pub enum AS {
    Immediate = 0,
    Native = 4,
}

impl AS {
    fn to_field<F: PrimeField64>(self) -> F {
        match self {
            AS::Immediate => F::ZERO,
            AS::Native => F::from_canonical_u8(AS::Native as u8),
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

/// Warning: for extension field branch instructions, the `pc, labels` **must** be using
/// `DEFAULT_PC_STEP`.
fn convert_instruction<F: PrimeField32, EF: ExtensionField<F>>(
    instruction: AsmInstruction<F, EF>,
    debug_info: Option<DebugInfo>,
    pc: F,
    labels: impl Fn(F) -> F,
    options: &CompilerOptions,
) -> Program<F> {
    let instructions = match instruction {
        AsmInstruction::LoadFI(dst, src, index, size, offset) => vec![
            // mem[dst] <- mem[mem[src] + index * size + offset]
            inst(
                options.opcode_with_offset(NativeLoadStoreOpcode::LOADW),
                i32_f(dst),
                index * size + offset,
                i32_f(src),
                AS::Native,
                AS::Native,
            ),
        ],
        AsmInstruction::LoadEI(dst, src, index, size, offset) => vec![
            // mem[dst] <- mem[mem[src] + index * size + offset]
            inst(
                options.opcode_with_offset(NativeLoadStore4Opcode(NativeLoadStoreOpcode::LOADW)),
                i32_f(dst),
                index * size + offset,
                i32_f(src),
                AS::Native,
                AS::Native,
            ),
        ],
        AsmInstruction::StoreFI(val, addr, index, size, offset) => vec![
            // mem[mem[addr] + index * size + offset] <- mem[val]
            inst(
                options.opcode_with_offset(NativeLoadStoreOpcode::STOREW),
                i32_f(val),
                index * size + offset,
                i32_f(addr),
                AS::Native,
                AS::Native,
            ),
        ],
        AsmInstruction::StoreEI(val, addr, index, size, offset) => vec![
            // mem[mem[addr] + index * size + offset] <- mem[val]
            inst(
                options.opcode_with_offset(NativeLoadStore4Opcode(NativeLoadStoreOpcode::STOREW)),
                i32_f(val),
                index * size + offset,
                i32_f(addr),
                AS::Native,
                AS::Native,
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
                    AS::Native,
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
                AS::Native,
                AS::Native,
            ),
        ],
        AsmInstruction::BneI(label, lhs, rhs) => vec![
            // if mem[lhs] != rhs, pc <- labels[label]
            inst(
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BNE)),
                i32_f(lhs),
                rhs,
                labels(label) - pc,
                AS::Native,
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
                AS::Native,
                AS::Native,
            ),
        ],
        AsmInstruction::BeqI(label, lhs, rhs) => vec![
            // if mem[lhs] == rhs, pc <- labels[label]
            inst(
                options.opcode_with_offset(NativeBranchEqualOpcode(BranchEqualOpcode::BEQ)),
                i32_f(lhs),
                rhs,
                labels(label) - pc,
                AS::Native,
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
                AS::Native,
                AS::Native,
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
                AS::Native,
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
                AS::Native,
                AS::Native,
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
                AS::Native,
                AS::Immediate,
            ))
            .collect(),
        AsmInstruction::Trap => vec![
            Instruction::phantom(PhantomDiscriminant(SysPhantom::DebugPanic as u16), F::ZERO, F::ZERO, 0),
            // Ensure that the program terminates unsuccessfully.
            inst(
                options.opcode_with_offset(SystemOpcode::TERMINATE),
                F::ZERO,
                F::ZERO,
                F::ONE,
                AS::Immediate,
                AS::Immediate,
            ),
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
        AsmInstruction::HintFelt() => vec![
            Instruction::phantom(PhantomDiscriminant(NativePhantom::HintFelt as u16), F::ZERO, F::ZERO, 0)
        ],
        AsmInstruction::HintBits(src, len) => vec![
            Instruction::phantom(PhantomDiscriminant(NativePhantom::HintBits as u16), i32_f(src), F::from_canonical_u32(len), AS::Native as u16)
        ],
        AsmInstruction::HintLoad() => vec![
            Instruction::phantom(PhantomDiscriminant(NativePhantom::HintLoad as u16), F::ZERO, F::ZERO, 0)
        ],
        AsmInstruction::StoreHintWordI(val, offset) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::HINT_STOREW),
            F::ZERO,
            offset,
            i32_f(val),
            AS::Native,
            AS::Native,
        )],
        AsmInstruction::StoreHintExtI(val, offset) => vec![inst(
            options.opcode_with_offset(NativeLoadStore4Opcode(NativeLoadStoreOpcode::HINT_STOREW)),
            F::ZERO,
            offset,
            i32_f(val),
            AS::Native,
            AS::Native,
        )],
        AsmInstruction::PrintV(src) => vec![Instruction::phantom(
            PhantomDiscriminant(NativePhantom::Print as u16),
            i32_f(src),
            F::ZERO,
            AS::Native as u16,
        )],
        AsmInstruction::PrintF(src) => vec![Instruction::phantom(
            PhantomDiscriminant(NativePhantom::Print as u16),
            i32_f(src),
            F::ZERO,
            AS::Native as u16,
        )],
        AsmInstruction::PrintE(src) => (0..EF::D as i32)
            .map(|i| {
                Instruction::phantom(
                    PhantomDiscriminant(NativePhantom::Print as u16),
                    i32_f(src + i),
                    F::ZERO,
                    AS::Native as u16,
                )
            })
            .collect(),
        AsmInstruction::ImmF(dst, val) =>
            vec![inst_med(
                options.opcode_with_offset(FieldArithmeticOpcode::ADD),
                i32_f(dst),
                val,
                F::ZERO,
                AS::Native,
                AS::Immediate,
                AS::Native,
            )],
        AsmInstruction::CopyF(dst, src) =>
            vec![inst_med(
                options.opcode_with_offset(FieldArithmeticOpcode::ADD),
                i32_f(dst),
                i32_f(src),
                F::ZERO,
                AS::Native,
                AS::Native,
                AS::Immediate
            )],
            AsmInstruction::AddF(dst, lhs, rhs) | AsmInstruction::SubF(dst, lhs, rhs) | AsmInstruction::MulF(dst, lhs, rhs) | AsmInstruction::DivF(dst, lhs, rhs) => vec![
                // AddF: mem[dst] <- mem[lhs] + mem[rhs]
                // SubF: mem[dst] <- mem[lhs] - mem[rhs]
                // MulF: mem[dst] <- mem[lhs] * mem[rhs]
                // DivF: mem[dst] <- mem[lhs] / mem[rhs]
                inst_med(
                    match instruction {
                        AsmInstruction::AddF(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::ADD),
                        AsmInstruction::SubF(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::SUB),
                        AsmInstruction::MulF(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::MUL),
                        AsmInstruction::DivF(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::DIV),
                        _ => unreachable!(),
                    },
                    i32_f(dst),
                    i32_f(lhs),
                    i32_f(rhs),
                    AS::Native,
                    AS::Native,
                    AS::Native,
                ),
            ],
            AsmInstruction::AddFI(dst, lhs, rhs) | AsmInstruction::SubFI(dst, lhs, rhs) | AsmInstruction::MulFI(dst, lhs, rhs) | AsmInstruction::DivFI(dst, lhs, rhs) => vec![
                // AddFI: mem[dst] <- mem[lhs] + rhs
                // SubFI: mem[dst] <- mem[lhs] - rhs
                // MulFI: mem[dst] <- mem[lhs] * rhs
                // DivFI: mem[dst] <- mem[lhs] / rhs
                inst_med(
                    match instruction {
                        AsmInstruction::AddFI(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::ADD),
                        AsmInstruction::SubFI(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::SUB),
                        AsmInstruction::MulFI(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::MUL),
                        AsmInstruction::DivFI(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::DIV),
                        _ => unreachable!(),
                    },
                    i32_f(dst),
                    i32_f(lhs),
                    rhs,
                    AS::Native,
                    AS::Native,
                    AS::Immediate,
                ),
            ],
            AsmInstruction::SubFIN(dst, lhs, rhs) | AsmInstruction::DivFIN(dst, lhs, rhs) => vec![
                // SubFIN: mem[dst] <- lhs - mem[rhs]
                // DivFIN: mem[dst] <- lhs / mem[rhs]
                inst_med(
                    match instruction {
                        AsmInstruction::SubFIN(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::SUB),
                        AsmInstruction::DivFIN(_, _, _) => options.opcode_with_offset(FieldArithmeticOpcode::DIV),
                        _ => unreachable!(),
                    },
                    i32_f(dst),
                    lhs,
                    i32_f(rhs),
                    AS::Native,
                    AS::Immediate,
                    AS::Native,
                ),
            ],
            AsmInstruction::AddE(dst, lhs, rhs) | AsmInstruction::SubE(dst, lhs, rhs) | AsmInstruction::MulE(dst, lhs, rhs) | AsmInstruction::DivE(dst, lhs, rhs) => vec![
                // AddE: mem[dst] <- mem[lhs] + mem[rhs]
                // SubE: mem[dst] <- mem[lhs] - mem[rhs]
                inst(
                    match instruction {
                        AsmInstruction::AddE(_, _, _) => options.opcode_with_offset(FieldExtensionOpcode::FE4ADD),
                        AsmInstruction::SubE(_, _, _) => options.opcode_with_offset(FieldExtensionOpcode::FE4SUB),
                        AsmInstruction::MulE(_, _, _) => options.opcode_with_offset(FieldExtensionOpcode::BBE4MUL),
                        AsmInstruction::DivE(_, _, _) => options.opcode_with_offset(FieldExtensionOpcode::BBE4DIV),
                        _ => unreachable!(),
                    },
                    i32_f(dst),
                    i32_f(lhs),
                    i32_f(rhs),
                    AS::Native,
                    AS::Native,
            )],
        AsmInstruction::Poseidon2Compress(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(Poseidon2Opcode::COMP_POS2),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Native,
            AS::Native,
        )],
        AsmInstruction::Poseidon2Permute(dst, src) => vec![inst(
            options.opcode_with_offset(Poseidon2Opcode::PERM_POS2),
            i32_f(dst),
            i32_f(src),
            F::ZERO,
            AS::Native,
            AS::Native,
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
            AS::Native,
            AS::Native,
        )],
        AsmInstruction::FriReducedOpening(a, b, length, alpha, res, hint_id, is_init) => vec![Instruction {
            opcode: options.opcode_with_offset(FriOpcode::FRI_REDUCED_OPENING),
            a: i32_f(a),
            b: i32_f(b),
            c: i32_f(length),
            d: i32_f(alpha),
            e: i32_f(res),
            f: i32_f(hint_id),
            g: i32_f(is_init),
        }],
        AsmInstruction::VerifyBatchFelt(dim, opened, opened_length, sibling, index, commit) => vec![Instruction {
            opcode: options.opcode_with_offset(VerifyBatchOpcode::VERIFY_BATCH),
            a: i32_f(dim),
            b: i32_f(opened),
            c: i32_f(opened_length),
            d: i32_f(sibling),
            e: i32_f(index),
            f: i32_f(commit),
            g: F::ONE,
        }],
        AsmInstruction::VerifyBatchExt(dim, opened, opened_length, sibling, index, commit) => vec![Instruction {
            opcode: options.opcode_with_offset(VerifyBatchOpcode::VERIFY_BATCH),
            a: i32_f(dim),
            b: i32_f(opened),
            c: i32_f(opened_length),
            d: i32_f(sibling),
            e: i32_f(index),
            f: i32_f(commit),
            g: F::from_canonical_usize(4).inverse(),
        }],
        AsmInstruction::RangeCheck(v, x_bit, y_bit) => {
            assert!((0..=16).contains(&x_bit));
            assert!((0..=14).contains(&y_bit));
            vec!
            [inst(
                options.opcode_with_offset(NativeRangeCheckOpcode::RANGE_CHECK),
                i32_f(v),
                i32_f(x_bit),
                i32_f(y_bit),
                AS::Native,
                // Here it just requires a 0
                AS::Immediate,
            )]
        }
    };

    let debug_infos = vec![debug_info; instructions.len()];

    Program::from_instructions_and_debug_infos(&instructions, &debug_infos)
}

pub fn convert_program<F: PrimeField32, EF: ExtensionField<F>>(
    program: AssemblyCode<F, EF>,
    options: CompilerOptions,
) -> Program<F> {
    // mem[0] <- 0
    let init_register_0 = inst_med(
        options.opcode_with_offset(FieldArithmeticOpcode::ADD),
        F::ZERO,
        F::ZERO,
        F::ZERO,
        AS::Native,
        AS::Immediate,
        AS::Immediate,
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

    let mut result = Program::new_empty(0);
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
