use p3_field::{ExtensionField, PrimeField32, TwoAdicField};
use stark_vm::program::Program;

use super::{config::AsmConfig, AsmCompiler};
use crate::{
    conversion::{convert_program, CompilerOptions},
    prelude::Builder,
};

/// A builder that compiles assembly code.
pub type AsmBuilder<F, EF> = Builder<AsmConfig<F, EF>>;

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> AsmBuilder<F, EF> {
    pub fn compile_isa<const WORD_SIZE: usize>(self) -> Program<F> {
        self.compile_isa_with_options::<WORD_SIZE>(CompilerOptions::default())
    }

    pub fn compile_isa_with_options<const WORD_SIZE: usize>(
        self,
        options: CompilerOptions,
    ) -> Program<F> {
        let mut compiler = AsmCompiler::new(WORD_SIZE);
        compiler.build(self.operations);
        let asm_code = compiler.code();
        convert_program::<WORD_SIZE, F, EF>(asm_code, options)
    }
}
