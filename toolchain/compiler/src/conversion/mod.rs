use std::collections::HashMap;

use axvm_instructions::instruction::{DebugInfo, Instruction};
use num_bigint_dig::BigUint;
use p3_field::{ExtensionField, PrimeField32, PrimeField64};
use program::DEFAULT_PC_STEP;
use stark_vm::arch::{
    instructions::{program::Program, *},
    Modulus,
};
use strum::EnumCount;

use crate::asm::{AsmInstruction, AssemblyCode};

#[derive(Clone, Debug)]
pub struct CompilerOptions {
    // The compiler will ensure that the heap pointer is aligned to be a multiple of `word_size`.
    pub word_size: usize,
    pub compile_prints: bool,
    pub enable_cycle_tracker: bool,
    pub field_arithmetic_enabled: bool,
    pub field_extension_enabled: bool,
    pub enabled_modulus: Vec<BigUint>,
}

impl Default for CompilerOptions {
    fn default() -> Self {
        CompilerOptions {
            word_size: 8,
            compile_prints: true,
            enable_cycle_tracker: false,
            field_arithmetic_enabled: true,
            field_extension_enabled: true,
            enabled_modulus: Modulus::all().iter().map(|m| m.prime()).collect(),
        }
    }
}

impl CompilerOptions {
    pub fn opcode_with_offset<Opcode: UsizeOpcode>(&self, opcode: Opcode) -> usize {
        let offset = Opcode::default_offset();
        offset + opcode.as_usize()
    }

    pub fn modular_opcode_with_offset<Opcode: UsizeOpcode>(
        &self,
        opcode: Opcode,
        modulus: BigUint,
    ) -> usize {
        let res = self.opcode_with_offset(opcode);
        let modulus_id = self
            .enabled_modulus
            .iter()
            .position(|m| m == &modulus)
            .unwrap_or_else(|| panic!("unsupported modulus: {}", modulus));
        let modular_shift = modulus_id * ModularArithmeticOpcode::COUNT;
        res + modular_shift
    }
}

fn inst<F: PrimeField64>(opcode: usize, a: F, b: F, c: F, d: AS, e: AS) -> Instruction<F> {
    Instruction {
        opcode,
        a,
        b,
        c,
        d: d.to_field(),
        e: e.to_field(),
        f: F::zero(),
        g: F::zero(),
        debug: String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_med<F: PrimeField64>(
    opcode: usize,
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
        g: F::zero(),
        debug: String::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn inst_large<F: PrimeField64>(
    opcode: usize,
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
        debug: String::new(),
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
    options: &CompilerOptions,
) -> Vec<Instruction<F>> {
    match instruction {
        AsmInstruction::EqU256(a, b, c) => vec![inst_large(
            options.opcode_with_offset(U256Opcode::EQ),
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
            PhantomInstruction::PrintF,
            i32_f(src),
            F::zero(),
            0,
        )],
        AsmInstruction::PrintF(src) => vec![Instruction::phantom(
            PhantomInstruction::PrintF,
            i32_f(src),
            F::zero(),
            0,
        )],
        AsmInstruction::PrintE(src) => (0..EF::D as i32)
            .map(|i| {
                Instruction::phantom(
                    PhantomInstruction::PrintF,
                    i32_f(src + i * word_size_i32),
                    F::zero(),
                    0,
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
                    F::zero(),
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
            Instruction::phantom(PhantomInstruction::DebugPanic, F::zero(), F::zero(), 0),
        ],
        AsmInstruction::Halt => vec![
            // terminate
            inst(
                options.opcode_with_offset(CommonOpcode::TERMINATE),
                F::zero(),
                F::zero(),
                F::zero(),
                AS::Immediate,
                AS::Immediate,
            ),
        ],
        AsmInstruction::HintInputVec() => vec![
            Instruction::phantom(PhantomInstruction::HintInput, F::zero(), F::zero(), 0)
        ],
        AsmInstruction::HintBits(src, len) => vec![
            Instruction::phantom(PhantomInstruction::HintBits, i32_f(src), F::from_canonical_u32(len), AS::Memory as u16)
        ],
        AsmInstruction::StoreHintWordI(val, offset) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::SHINTW),
            F::zero(),
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
            F::zero(),
            i32_f(dst),
            AS::Immediate,
            AS::Memory,
        )],
        AsmInstruction::CopyF(dst, src) => vec![inst(
            options.opcode_with_offset(NativeLoadStoreOpcode::LOADW),
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
                convert_base_arithmetic_instruction(instruction, options)
            } else {
                panic!(
                    "Unsupported instruction {:?}, field arithmetic is disabled",
                    instruction
                )
            }
        }
        AsmInstruction::EqU256(..) => convert_comparison_instruction(instruction, options),
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
            F::zero(),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ModularAdd(modulus, dst, src1, src2) => vec![inst(
            options.modular_opcode_with_offset(ModularArithmeticOpcode::ADD, modulus),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ModularSub(modulus, dst, src1, src2) => vec![inst(
            options.modular_opcode_with_offset(ModularArithmeticOpcode::SUB, modulus),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ModularMul(modulus, dst, src1, src2) => vec![inst(
            options.modular_opcode_with_offset(ModularArithmeticOpcode::MUL, modulus),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ModularDiv(modulus, dst, src1, src2) => vec![inst(
            options.modular_opcode_with_offset(ModularArithmeticOpcode::DIV, modulus),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Add256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::ADD),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Sub256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::SUB),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Mul256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::MUL),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::LessThanU256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::LT),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::EqualTo256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::EQ),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Xor256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::XOR),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::And256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::AND),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Or256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::OR),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::LessThanI256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::SLT),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ShiftLeft256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::SLL),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ShiftRightLogic256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::SRL),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::ShiftRightArith256(dst, src1, src2) => vec![inst(
            options.opcode_with_offset(U256Opcode::SRA),
            i32_f(dst),
            i32_f(src1),
            i32_f(src2),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Keccak256(dst, src, len) => vec![inst_med(
            options.opcode_with_offset(Keccak256Opcode::KECCAK256),
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
        AsmInstruction::Secp256k1AddUnequal(dst_ptr_ptr, p_ptr_ptr, q_ptr_ptr) => vec![inst_med(
            options.opcode_with_offset(EccOpcode::EC_ADD_NE),
            i32_f(dst_ptr_ptr),
            i32_f(p_ptr_ptr),
            i32_f(q_ptr_ptr),
            AS::Memory,
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::Secp256k1Double(dst_ptr_ptr, p_ptr_ptr) => vec![inst(
            options.opcode_with_offset(EccOpcode::EC_DOUBLE),
            i32_f(dst_ptr_ptr),
            i32_f(p_ptr_ptr),
            F::zero(),
            AS::Memory,
            AS::Memory,
        )],
        AsmInstruction::CycleTrackerStart(name) => {
            if options.enable_cycle_tracker {
                vec![Instruction::debug(PhantomInstruction::CtStart, &name)]
            } else {
                vec![]
            }
        }
        AsmInstruction::CycleTrackerEnd(name) => {
            if options.enable_cycle_tracker {
                vec![Instruction::debug(PhantomInstruction::CtEnd, &name)]
            } else {
                vec![]
            }
        }
        AsmInstruction::Publish(val, index) => vec![inst_med(
            options.opcode_with_offset(PublishOpcode::PUBLISH),
            F::zero(),
            i32_f(val),
            i32_f(index),
            AS::Immediate,
            AS::Memory,
            AS::Memory,
        )],
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
        F::zero(),
        F::zero(),
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

    let mut instructions_and_debug_infos = HashMap::new();
    instructions_and_debug_infos.insert(0, (init_register_0, init_debug_info));
    for block in program.blocks.iter() {
        for (instruction, debug_info) in block.0.iter().zip(block.1.iter()) {
            let cur_size = instructions_and_debug_infos.len() as u32;
            let cur_pc = cur_size * DEFAULT_PC_STEP;

            let labels =
                |label: F| F::from_canonical_u32(block_start[label.as_canonical_u64() as usize]);
            let result = convert_instruction(
                instruction.clone(),
                debug_info.clone(),
                F::from_canonical_u32(cur_pc),
                labels,
                &options,
            );

            for (local_pc, (instruction, debug_info)) in result.instructions_and_debug_infos {
                let existing = instructions_and_debug_infos
                    .insert(cur_pc + local_pc, (instruction.clone(), debug_info.clone()));
                assert!(existing.is_none(), "pc should not already exist");
            }
        }
    }

    Program {
        instructions_and_debug_infos,
        step: DEFAULT_PC_STEP,
        pc_start: 0,
        pc_base: 0,
    }
}
