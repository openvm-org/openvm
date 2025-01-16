use std::{fs::File, io::Write};

use openvm_native_compiler::{
    asm::{AsmBuilder, AsmCompiler},
    conversion::{convert_program, CompilerOptions},
    ir::{Felt, Var},
};
use openvm_native_transpiler::serialize_defined_instructions;
use openvm_stark_backend::p3_field::{extension::BinomialExtensionField, FieldAlgebra};
use openvm_stark_sdk::p3_baby_bear::BabyBear;

type F = BabyBear;
type EF = BinomialExtensionField<BabyBear, 4>;

fn function_name(builder: &mut AsmBuilder<F, EF>, n: Var<F>) -> Felt<F> {
    let a: Felt<_> = builder.constant(F::ZERO);
    let b: Felt<_> = builder.constant(F::ONE);
    let zero: Var<_> = builder.eval(F::ZERO);
    builder.range(zero, n).for_each(|_, builder| {
        builder.assign(&b, a + b);
        builder.assign(&a, b - a);
    });
    a
}

fn main() {
    let mut file = File::create("compiler_output.txt").unwrap();

    let mut builder = AsmBuilder::<F, EF>::default();

    let var_n = builder.uninit();
    let result = function_name(&mut builder, var_n);

    writeln!(file, "{}", var_n.fp()).unwrap();
    writeln!(file, "{}", result.fp()).unwrap();

    let mut compiler = AsmCompiler::new(1);
    compiler.build(builder.operations);
    let asm_code = compiler.code();
    let program = convert_program::<F, EF>(asm_code, CompilerOptions::default());
    let serialized = serialize_defined_instructions(&program.defined_instructions());
    for word in serialized {
        writeln!(file, "{}", word).unwrap();
    }
}
