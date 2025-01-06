use crate::parse_kernel::ParsedKernel;

const MAIN_PLACEHOLDER: &str = "<...>";
const OUTSIDE_PLACEHOLDER: &str = "<===>";
const EDSL_TEMPLATE: &str = "
use openvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{convert_program, CompilerOptions},
    ir::{Array, Ext, Felt},
};
use p3_baby_bear::BabyBear;
use p3_field::{extension::BinomialExtensionField, AbstractField};

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

<===>

fn main() {
    let mut builder = AsmBuilder::<F, EF>::default();

<...>

    let mut compiler = AsmCompiler::new(1);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    println!(\"{}\", program);a stru
}
";

pub const CONFIG: &str = "
[package]
name = \"root\"
version = \"0.1.0\"
edition = \"2021\"

[dependencies]
openvm-native-compiler = { path = \"../../../../extensions/native/compiler\" }
p3-field = { git = \"https://github.com/Plonky3/Plonky3.git\", rev = \"9b267c4\" }
p3-baby-bear = { git = \"https://github.com/Plonky3/Plonky3.git\", features = [
\"nightly-features\",
], rev = \"9b267c4\" }

[workspace]
";

fn produce_edsl_function(parsed_kernel: &ParsedKernel) -> String {
    let mut result = String::new();

    result.push_str("fn edsl_function(builder: &mut AsmBuilder<F, EF>");
    for argument in &parsed_kernel.arguments {
        result.push_str(", ");
        result.push_str(&argument.name);
        result.push_str(": ");
        result.push_str(&argument.edsl_type);
    }
    result.push_str(") -> ");
    result.push_str(&parsed_kernel.edsl_return_type);
    result.push_str(" {\n");
    result.push_str(&parsed_kernel.body);
    result.push_str("\n}");

    result
}

fn produce_main_portion(parsed_kernel: &ParsedKernel) -> String {
    let mut result = String::new();

    for argument in &parsed_kernel.arguments {
        result.push_str(&format!(
            "\tlet var_{}: {} = builder.uninit();\n",
            argument.name, argument.edsl_type
        ));
        result.push_str(&format!(
            "\tprintln!(\"{{}}\", var_{}.fp());\n",
            argument.name
        ));
    }

    result.push_str("\tlet result = edsl_function(&mut builder");
    for argument in &parsed_kernel.arguments {
        result.push_str(&format!(", var_{}", argument.name));
    }
    result.push_str(");\n");
    result.push_str("\tprintln!(\"{}\", result.fp());\n");

    result
}

pub fn produce_edsl_code(parsed_kernel: &ParsedKernel) -> String {
    let mut result = EDSL_TEMPLATE.to_string();

    result = result.replace(MAIN_PLACEHOLDER, &produce_main_portion(parsed_kernel));
    result = result.replace(OUTSIDE_PLACEHOLDER, &produce_edsl_function(parsed_kernel));

    result
}
