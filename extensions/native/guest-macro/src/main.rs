/*use axvm_edsl_kernel::{
    execute::execute_edsl,
    generate_edsl::{produce_edsl_code, CONFIG},
    parse_compiler_output::{parse_compiled_kernel, CompiledKernel},
    parse_kernel::parse_raw_kernel,
    transportation::compiled_kernel_to_function,
};*/
use p3_baby_bear::BabyBear;
use quote::quote;

pub fn main() {
    //println!("Hello, world!");

    /*let kernel_source = quote! {
        fn function_name(foo: usize | Felt<F>, bar: usize | Felt<F>) -> usize | Felt<F> {
            return builder.eval(foo + bar);
        }
    };

    let parsed_kernel = parse_raw_kernel(kernel_source);

    //println!("{:?}", parse_raw_kernel(kernel_source));

    /*let parsed_kernel = ParsedKernel {
            function_name: "function_name".to_string(),
            arguments: vec![
                KernelArgument {
                    name: "foo".to_string(),
                    rust_type: "usize".to_string(),
                    edsl_type: "Felt<F>".to_string(),
                },
                KernelArgument {
                    name: "bar".to_string(),
                    rust_type: "usize".to_string(),
                    edsl_type: "Felt<F>".to_string(),
                },
            ],
            body: "
    return builder.eval(foo + bar);
            ".to_string(),
            rust_return_type: "usize".to_string(),
            edsl_return_type: "Felt<F>".to_string(),
        };*/
    let edsl_code = produce_edsl_code(&parsed_kernel);
    let config = CONFIG.to_string();
    let compiler_output = execute_edsl(edsl_code, config).unwrap();
    println!("compiler_output = {}", compiler_output);

    let compiled_kernel: CompiledKernel<BabyBear> =
        parse_compiled_kernel(parsed_kernel, compiler_output);
    println!("compiled_kernel = {:?}", compiled_kernel);

    let rust_function = compiled_kernel_to_function(compiled_kernel);
    println!("rust_function = {}", rust_function);*/
}
