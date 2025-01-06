use p3_baby_bear::BabyBear;

use crate::parse_compiler_output::CompiledKernel;

mod parse_compiler_output;
mod parse_kernel;
mod transportation;
const UTILITY_CELL: usize = 0;
const PC_STEP: usize = 4;
const REGISTER_LIMBS: usize = 4;
const REGISTER_LIMB_SIZE: usize = 256;

#[proc_macro]
pub fn edsl_kernel(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = proc_macro2::TokenStream::from(input);

    let parsed_kernel = parse_kernel::parse_raw_kernel(input);
    let compiler_output = std::fs::read_to_string(parsed_kernel.file_path.clone()).unwrap();
    let compiled_kernel: CompiledKernel<BabyBear> =
        parse_compiler_output::parse_compiled_kernel(parsed_kernel, compiler_output);
    let rust_function = transportation::compiled_kernel_to_function(compiled_kernel);

    std::fs::write("macro_output.txt", rust_function.clone()).unwrap();

    let output: proc_macro2::TokenStream = rust_function.parse().unwrap();

    proc_macro::TokenStream::from(output)
}
