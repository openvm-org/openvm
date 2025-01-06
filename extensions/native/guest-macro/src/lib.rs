use p3_baby_bear::BabyBear;

use crate::parse_compiler_output::CompiledKernel;

mod execute;
mod generate_edsl;
mod parse_compiler_output;
mod parse_kernel;
mod transportation;

const IMMEDIATE_ADDRESS_SPACE: usize = 0;
const RUST_REGISTER_ADDRESS_SPACE: usize = 1;
const KERNEL_ADDRESS_SPACE: usize = 5;
const LONG_FORM_INSTRUCTION_INDICATOR: u32 = (1 << 31) + 115115115;
const GAP_INDICATOR: u32 = (1 << 31) + 113113113;
const UTILITY_CELL: usize = 0;
const VARIABLE_REGISTER_INDICATOR: u32 = (1 << 31) + 116;
const PC_STEP: usize = 4;
const REGISTER_LIMBS: usize = 4;
const REGISTER_LIMB_SIZE: usize = 256;

#[proc_macro]
pub fn edsl_kernel(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input);

    let parsed_kernel = parse_kernel::parse_raw_kernel(input);
    let edsl_code = generate_edsl::produce_edsl_code(&parsed_kernel);
    let config = generate_edsl::CONFIG.to_string();
    let compiler_output = execute::execute_edsl(edsl_code, config).unwrap();
    let compiled_kernel: CompiledKernel<BabyBear> =
        parse_compiler_output::parse_compiled_kernel(parsed_kernel, compiler_output);
    let rust_function = transportation::compiled_kernel_to_function(compiled_kernel);

    std::fs::write("macro_output.txt", rust_function.clone()).unwrap();

    let output: proc_macro2::TokenStream = rust_function.parse().unwrap();

    proc_macro::TokenStream::from(output)
}
