use openvm_instructions::instruction::Instruction;
use openvm_native_transpiler::deserialize_defined_instructions;
use p3_field::{Field, PrimeField32};

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

pub fn parse_compiled_kernel<F: PrimeField32>(
    parsed_kernel: ParsedKernel,
    compiler_output: String,
) -> CompiledKernel<F> {
    let words: Vec<u32> = compiler_output
        .lines()
        .map(|line| line.parse().unwrap())
        .collect();
    let mut index = 0;

    let arguments = parsed_kernel
        .arguments
        .into_iter()
        .map(|argument| {
            let name = argument.name;
            let rust_type = argument.rust_type;
            let edsl_type = argument.edsl_type;
            let fp = words[index] as usize;
            index += 1;
            CompiledKernelArgument {
                name,
                rust_type,
                edsl_type,
                fp,
            }
        })
        .collect::<Vec<_>>();
    let return_fp = words[index] as usize;
    index += 1;

    let instructions = deserialize_defined_instructions(&words[index..]);

    CompiledKernel {
        function_name: parsed_kernel.function_name,
        arguments,
        body: instructions,
        rust_return_type: parsed_kernel.rust_return_type,
        edsl_return_type: parsed_kernel.edsl_return_type,
        return_fp,
    }
}
