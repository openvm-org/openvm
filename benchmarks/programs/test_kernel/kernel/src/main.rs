use std::fs::File;
use p3_baby_bear::BabyBear;
use p3_field::extension::BinomialExtensionField;
use std::io::Write;

use openvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{CompilerOptions, convert_program}
    ,
};
use openvm_native_compiler::ir::Felt;
use openvm_native_serialization::serialize_instructions;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

fn function_name(builder: &mut AsmBuilder<F, EF>, foo: Felt<F>, bar: Felt<F>) -> Felt<F> {
    return builder.eval(foo + bar);
}

fn main() {
    let mut file = File::create("../compiler_output.txt").unwrap();

    let mut builder = AsmBuilder::<F, EF>::default();

    let var_foo = builder.uninit();
    let var_bar = builder.uninit();
    let result = function_name(&mut builder, var_foo, var_bar);

    writeln!(file, "{}", var_foo.fp()).unwrap();
    writeln!(file, "{}", var_bar.fp()).unwrap();
    writeln!(file, "{}", result.fp()).unwrap();

    let mut compiler = AsmCompiler::new(1);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    let serialized = serialize_instructions(&program.instructions());
    for word in serialized {
        writeln!(file, "{}", word).unwrap();
    }
}