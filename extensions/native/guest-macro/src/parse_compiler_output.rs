use openvm_instructions::{instruction::Instruction, VmOpcode};
use p3_field::Field;

use crate::parse_kernel::ParsedKernel;

#[derive(Debug)]
pub struct CompiledKernelArgument {
    pub name: String,
    pub rust_type: String,
    pub edsl_type: String,
    pub fp: usize,
}

#[derive(Debug)]
pub struct CompiledKernel<F: Field> {
    pub function_name: String,
    pub arguments: Vec<CompiledKernelArgument>,
    pub body: Vec<Instruction<F>>,
    pub rust_return_type: String,
    pub edsl_return_type: String,
    pub return_fp: usize,
}

pub fn parse_compiled_kernel<F: Field>(
    parsed_kernel: ParsedKernel,
    compiler_output: String,
) -> CompiledKernel<F> {
    let mut lines = compiler_output.lines();
    let arguments = parsed_kernel
        .arguments
        .into_iter()
        .map(|argument| {
            let name = argument.name;
            let rust_type = argument.rust_type;
            let edsl_type = argument.edsl_type;
            let fp = lines.next().unwrap().parse::<usize>().unwrap();
            CompiledKernelArgument {
                name,
                rust_type,
                edsl_type,
                fp,
            }
        })
        .collect::<Vec<_>>();
    let return_fp = lines.next().unwrap().parse::<usize>().unwrap();
    let instructions = lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let opcode = tokens[0][tokens[0].find('(').unwrap() + 1..tokens[0].len() - 1]
                .parse::<usize>()
                .unwrap();
            let operands = tokens[1..]
                .iter()
                .map(|token| F::from_canonical_usize(token.parse::<usize>().unwrap()))
                .collect::<Vec<_>>();
            Instruction {
                opcode: VmOpcode::from_usize(opcode),
                a: operands[0],
                b: operands[1],
                c: operands[2],
                d: operands[3],
                e: operands[4],
                f: operands[5],
                g: operands[6],
            }
        })
        .collect();

    CompiledKernel {
        function_name: parsed_kernel.function_name,
        arguments,
        body: instructions,
        rust_return_type: parsed_kernel.rust_return_type,
        edsl_return_type: parsed_kernel.edsl_return_type,
        return_fp,
    }
}
