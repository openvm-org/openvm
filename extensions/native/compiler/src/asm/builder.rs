use openvm_circuit::arch::instructions::program::Program;
use openvm_stark_backend::p3_field::{ExtensionField, PrimeField32, TwoAdicField};

use super::{config::AsmConfig, AsmCompiler};
use crate::{
    conversion::{convert_program, CompilerOptions},
    prelude::Builder,
};

/// A builder that compiles assembly code.
pub type AsmBuilder<F, EF> = Builder<AsmConfig<F, EF>>;

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> AsmBuilder<F, EF> {
    pub fn compile_isa(self) -> Program<F> {
        self.compile_isa_with_options(CompilerOptions::default())
    }

    pub fn compile_isa_with_options(self, options: CompilerOptions) -> Program<F> {
        let mut compiler = AsmCompiler::new(options.word_size);
        compiler.build(self.operations);
        let asm_code = compiler.code();
        convert_program(asm_code, options)
    }
}
