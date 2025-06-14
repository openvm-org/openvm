use openvm_circuit::arch::instructions::{
    instruction::{Instruction, NUM_OPERANDS},
    program::Program,
    LocalOpcode,
};
use openvm_continuations::F;
use openvm_native_compiler::{asm::A0, conversion::AS, NativeJalOpcode};
use openvm_stark_backend::p3_field::{FieldAlgebra, PrimeField32};
use rrs_lib::instruction_formats::IType;

const OPCODE: u32 = 0x0b;
const FUNCT3: u32 = 0b111;
const LONG_FORM_INSTRUCTION_INDICATOR: u32 = (FUNCT3 << 12) + OPCODE;
const GAP_INDICATOR: u32 = (1 << 25) + (FUNCT3 << 12) + OPCODE;

pub fn program_to_asm(mut program: Program<F>) -> String {
    let pc_diff = handle_pc_diff(&mut program);
    let assembly_and_comments = convert_program_to_u32s_and_comments(&program, pc_diff);
    let mut asm_output = String::new();
    for (u32s, comment) in &assembly_and_comments {
        for (idx, x) in u32s.iter().enumerate() {
            asm_output.push_str(&u32_to_directive(*x));
            if idx == 0 {
                asm_output.push_str(" // ");
                asm_output.push_str(comment);
            }
            asm_output.push('\n');
        }
    }
    asm_output
}

fn u32_to_directive(x: u32) -> String {
    let opcode = x & 0b1111111;
    let dec_insn = IType::new(x);
    format!(
        ".insn i {}, {}, x{}, x{}, {}",
        opcode, dec_insn.funct3, dec_insn.rd, dec_insn.rs1, dec_insn.imm
    )
}

/// In order to use native instructions in kernel functions, native instructions need to be
/// converted to RISC-V machine code(long form instructions) first. Then Rust compiler compiles the
/// whole program into an ELF. Finally, the ELF is transpiled into an OpenVm Exe.
/// In the perspective of the native compiler and the transpiler, the PC step between 2 native
/// instructions is 4. However, in the ELF, each native instruction takes longer than 4 bytes, so
/// the instructions after the code blocks use the actual lengths of the native instructions to
/// compute PC offsets. To solve this problem, we need the gap indicator to pad the native code
/// block in order to align the PC of the following instructions.
/// More details about long form instructions and gap indicators can be found in
/// `docs/specs/transpiler.md`.
fn handle_pc_diff(program: &mut Program<F>) -> usize {
    const GAP_INDICATOR_WIDTH: usize = 2;
    const LONG_FORM_NATIVE_INSTRUCTION_WIDTH: usize = 10;
    const PC_STEP: usize = 4;
    // For GAP_INDICATOR, whose width is 2.
    let mut pc_diff = GAP_INDICATOR_WIDTH;
    // For each native instruction
    pc_diff += program.num_defined_instructions() * (LONG_FORM_NATIVE_INSTRUCTION_WIDTH - 1);
    // For next jal
    pc_diff += LONG_FORM_NATIVE_INSTRUCTION_WIDTH - 1;
    let jal = Instruction::<F> {
        opcode: NativeJalOpcode::JAL.global_opcode(),
        a: F::from_canonical_usize(A0 as usize), // A0
        // +1 means the next instruction after the gap
        b: F::from_canonical_usize(PC_STEP * (pc_diff + 1)),
        c: F::from_canonical_usize(0),
        d: F::from_canonical_u32(AS::Native as u32),
        e: F::from_canonical_usize(0),
        f: F::from_canonical_usize(0),
        g: F::from_canonical_usize(0),
    };
    program.push_instruction(jal);
    pc_diff
}

fn convert_program_to_u32s_and_comments(
    program: &Program<F>,
    pc_diff: usize,
) -> Vec<(Vec<u32>, String)> {
    program
        .defined_instructions()
        .iter()
        .map(|ins| {
            (
                vec![
                    LONG_FORM_INSTRUCTION_INDICATOR,
                    NUM_OPERANDS as u32,
                    ins.opcode.as_usize() as u32,
                    ins.a.as_canonical_u32(),
                    ins.b.as_canonical_u32(),
                    ins.c.as_canonical_u32(),
                    ins.d.as_canonical_u32(),
                    ins.e.as_canonical_u32(),
                    ins.f.as_canonical_u32(),
                    ins.g.as_canonical_u32(),
                ],
                format!("{:?}", ins.opcode),
            )
        })
        .chain(std::iter::once((
            vec![GAP_INDICATOR, pc_diff as u32],
            "GAP_INDICATOR".to_string(),
        )))
        .collect()
}
