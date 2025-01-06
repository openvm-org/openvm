use openvm_instructions::{instruction::Instruction, VmOpcode};
use openvm_native_compiler::{CastfOpcode, FieldArithmeticOpcode, NativeJalOpcode};
use p3_field::{Field, PrimeField32};

use crate::{
    parse_compiler_output::CompiledKernel,
    transportation::Operand::{Literal, Variable},
    GAP_INDICATOR, IMMEDIATE_ADDRESS_SPACE, KERNEL_ADDRESS_SPACE, LONG_FORM_INSTRUCTION_INDICATOR,
    PC_STEP, REGISTER_LIMBS, REGISTER_LIMB_SIZE, RUST_REGISTER_ADDRESS_SPACE, UTILITY_CELL,
    VARIABLE_REGISTER_INDICATOR,
};

#[derive(Clone, Debug)]
pub enum Operand<F: Field> {
    Literal(F),
    Variable(String, usize),
}

impl<F: Field> Operand<F> {
    pub fn usize(val: usize) -> Self {
        Literal(F::from_canonical_usize(val))
    }

    pub fn whatever() -> Self {
        Literal(F::ZERO)
    }
}

#[derive(Clone, Debug)]
pub struct MacroInstruction<F: Field> {
    pub opcode: VmOpcode,
    pub operands: Vec<Operand<F>>,
}

impl<F: Field> MacroInstruction<F> {
    fn literal(instruction: Instruction<F>) -> Self {
        Self {
            opcode: instruction.opcode,
            operands: vec![
                Literal(instruction.a),
                Literal(instruction.b),
                Literal(instruction.c),
                Literal(instruction.d),
                Literal(instruction.e),
                Literal(instruction.f),
                Literal(instruction.g),
            ],
        }
    }

    fn new<const N: usize>(opcode: VmOpcode, operands: [Operand<F>; N]) -> Self {
        Self {
            opcode,
            operands: operands.to_vec(),
        }
    }
}

/*

how things are going to go:

transportation will provide some MacroInstructions

the body will be converted to MacroInstructions

MacroInstructions are converted to an asm! call
 */

pub fn compiled_kernel_to_function<F: PrimeField32>(compiled_kernel: CompiledKernel<F>) -> String {
    let mut instructions = vec![];
    let mut input_vars = vec![];
    let return_name = "result".to_string();

    let mut result = "fn ".to_string();
    result.push_str(&compiled_kernel.function_name);
    result.push('(');

    for argument in compiled_kernel.arguments {
        let var_name = "var_".to_string() + &argument.name;
        result.push_str(&var_name);
        result.push_str(": ");
        result.push_str(&argument.rust_type);
        result.push_str(", ");

        input_vars.push(var_name);
        instructions.extend(transport_rust_to_edsl(
            argument.rust_type,
            argument.edsl_type,
            "var_".to_string() + &argument.name,
            argument.fp,
        ));
    }
    result.push_str(") -> ");
    result.push_str(&compiled_kernel.rust_return_type);
    result.push_str(" {\n");

    result.push_str("\tlet result: ");
    result.push_str(&compiled_kernel.rust_return_type);
    result.push_str(";\n");

    instructions.extend(
        compiled_kernel
            .body
            .into_iter()
            .map(MacroInstruction::literal),
    );
    instructions.extend(transport_edsl_to_rust(
        compiled_kernel.rust_return_type,
        compiled_kernel.edsl_return_type,
        return_name.clone(),
        compiled_kernel.return_fp,
    ));

    let mut instructions_string = String::new();
    for instruction in instructions.iter() {
        instructions_string.push_str(&format!("{:?}\n", instruction));
    }
    std::fs::write("instructions.txt", instructions_string).expect("Failed to write file");

    let asm_call = instructions_to_asm_call(instructions, input_vars, vec![return_name.clone()]);
    result.push_str(&asm_call);
    result.push('\t');
    result.push_str(&return_name);
    result.push('\n');
    result.push_str("}\n");

    result
}

fn u32_to_directive(x: u32) -> String {
    let opcode = x & 0b1111111;
    let funct3 = (x >> 12) & 0b111;
    let rd = (x >> 7) & 0b11111;
    let rs1 = (x >> 15) & 0b11111;
    let mut simm12 = (x >> 20) as i32;
    if simm12 >= 1 << 11 {
        simm12 -= 1 << 12;
    }
    format!(
        ".insn i {}, {}, x{}, x{}, {}",
        opcode, funct3, rd, rs1, simm12
    )
}

fn operand_to_directives<F: PrimeField32>(operand: Operand<F>) -> Vec<String> {
    match operand {
        Literal(x) => vec![u32_to_directive(x.as_canonical_u32())],
        Variable(var, offset) => vec![
            u32_to_directive(VARIABLE_REGISTER_INDICATOR),
            format!(".insn i 0, 0, {{{}}}, x0, {}", var, offset),
        ],
    }
}

fn instruction_to_directives<F: PrimeField32>(instruction: MacroInstruction<F>) -> Vec<String> {
    let mut directives = vec![];

    directives.push(u32_to_directive(LONG_FORM_INSTRUCTION_INDICATOR));
    directives.push(u32_to_directive(instruction.operands.len() as u32));
    directives.push(u32_to_directive(instruction.opcode.as_usize() as u32));
    for operand in instruction.operands {
        directives.extend(operand_to_directives(operand));
    }

    directives
}

pub fn instructions_to_asm_call<F: PrimeField32>(
    instructions: Vec<MacroInstruction<F>>,
    input_vars: Vec<String>,
    output_vars: Vec<String>,
) -> String {
    let mut result = String::new();
    result.push_str("\tunsafe {\n");
    result.push_str("\t\tcore::arch::asm!(\n");

    let mut add_directives = |directives: Vec<String>| {
        for directive in directives {
            result.push_str(&format!("\t\t\t\"{}\",\n", directive));
        }
        result.push('\n');
    };

    let mut pc_diff = 2;
    for instruction in instructions {
        let directives = instruction_to_directives(instruction);
        pc_diff += directives.len() - 1;
        add_directives(directives);
    }

    let mut jal_instruction: MacroInstruction<F> = MacroInstruction::new(
        VmOpcode::with_default_offset(NativeJalOpcode::JAL),
        [
            Operand::usize(UTILITY_CELL),
            Operand::whatever(),
            Operand::whatever(),
            Operand::usize(KERNEL_ADDRESS_SPACE),
        ],
    );
    let jal_example_directives = instruction_to_directives(jal_instruction.clone());
    pc_diff += jal_example_directives.len() - 1;

    jal_instruction.operands[1] = Operand::usize(PC_STEP * (pc_diff + 1));
    add_directives(instruction_to_directives(jal_instruction));

    add_directives(vec![
        u32_to_directive(GAP_INDICATOR),
        u32_to_directive(pc_diff as u32),
    ]);

    for input_var in input_vars {
        result.push_str(&format!("\t\t\t{} = in(reg) {},\n", input_var, input_var));
    }
    for output_var in output_vars {
        result.push_str(&format!(
            "\t\t\t{} = out(reg) {},\n",
            output_var, output_var
        ));
    }
    result.push_str("\t\t)\n");
    result.push_str("\t}\n");
    result
}

pub fn transport_rust_to_edsl<F: Field>(
    rust_type: String,
    edsl_type: String,
    rust_name: String,
    edsl_fp: usize,
) -> Vec<MacroInstruction<F>> {
    match (rust_type.as_str(), edsl_type.as_str()) {
        ("usize", "Felt<F>") => transport_usize_to_felt(rust_name, edsl_fp),
        _ => panic!(
            "Unsupported conversion from rust type {:?} to edsl type {:?}",
            rust_type, edsl_type
        ),
    }
}

fn transport_usize_to_felt<F: Field>(
    rust_name: String,
    edsl_fp: usize,
) -> Vec<MacroInstruction<F>> {
    let mut result = vec![];
    for i in (0..REGISTER_LIMBS).rev() {
        // add [{rust_name} + i] to [edsl_fp]
        result.push(MacroInstruction::new(
            VmOpcode::with_default_offset(FieldArithmeticOpcode::ADD),
            [
                Operand::usize(edsl_fp),
                Operand::usize(if i == REGISTER_LIMBS - 1 { 0 } else { edsl_fp }),
                Variable(rust_name.clone(), i),
                Operand::usize(KERNEL_ADDRESS_SPACE),
                Operand::usize(if i == REGISTER_LIMBS - 1 {
                    IMMEDIATE_ADDRESS_SPACE
                } else {
                    KERNEL_ADDRESS_SPACE
                }),
                Operand::usize(RUST_REGISTER_ADDRESS_SPACE),
            ],
        ));
        if i > 0 {
            result.push(MacroInstruction::new(
                VmOpcode::with_default_offset(FieldArithmeticOpcode::MUL),
                [
                    Operand::usize(edsl_fp),
                    Operand::usize(edsl_fp),
                    Operand::usize(REGISTER_LIMB_SIZE),
                    Operand::usize(KERNEL_ADDRESS_SPACE),
                    Operand::usize(KERNEL_ADDRESS_SPACE),
                    Operand::usize(IMMEDIATE_ADDRESS_SPACE),
                ],
            ));
        }
    }
    result
}

pub fn transport_edsl_to_rust<F: Field>(
    rust_type: String,
    edsl_type: String,
    rust_name: String,
    edsl_fp: usize,
) -> Vec<MacroInstruction<F>> {
    match (rust_type.as_str(), edsl_type.as_str()) {
        ("usize", "Felt < F >") => transport_felt_to_usize(rust_name, edsl_fp),
        _ => panic!(
            "Unsupported conversion from edsl type {:?} to rust type {:?}",
            edsl_type, rust_type,
        ),
    }
}

fn transport_felt_to_usize<F: Field>(
    rust_name: String,
    edsl_fp: usize,
) -> Vec<MacroInstruction<F>> {
    vec![MacroInstruction::new(
        VmOpcode::with_default_offset(CastfOpcode::CASTF),
        [
            Variable(rust_name, 0),
            Operand::usize(edsl_fp),
            Operand::usize(0),
            Operand::usize(RUST_REGISTER_ADDRESS_SPACE),
            Operand::usize(KERNEL_ADDRESS_SPACE),
        ],
    )]
}
