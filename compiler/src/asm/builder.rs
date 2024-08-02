use backtrace::Backtrace;

use p3_field::{ExtensionField, PrimeField32, TwoAdicField};

use stark_vm::cpu::trace::Instruction;

use crate::{
    conversion::{convert_program, CompilerOptions},
    ir::{Config, DslIr},
    prelude::Builder,
};

use super::{config::AsmConfig, AsmCompiler};

/// A builder that compiles assembly code.
pub type AsmBuilder<F, EF> = Builder<AsmConfig<F, EF>>;

#[derive(Debug, Clone, Default)]
pub struct DebugInfo<C: Config> {
    dsl_instruction: DslIr<C>,
    trace: Option<Backtrace>,
}

impl<C: Config> DebugInfo<C> {
    pub fn new(dsl_instruction: DslIr<C>, trace: Option<Backtrace>) -> Self {
        Self {
            dsl_instruction,
            trace,
        }
    }
}

pub struct Program<F, C: Config> {
    pub isa_instructions: Vec<Instruction<F>>,
    pub debug_info_vec: Vec<Option<DebugInfo<C>>>,
}

impl<F, C: Config> Program<F, C> {
    pub fn len(&self) -> usize {
        self.isa_instructions.len()
    }
}

impl<F: PrimeField32 + TwoAdicField, EF: ExtensionField<F> + TwoAdicField> AsmBuilder<F, EF> {
    pub fn compile_isa<const WORD_SIZE: usize>(self) -> Program<F, AsmConfig<F, EF>> {
        self.compile_isa_with_options::<WORD_SIZE>(CompilerOptions::default())
    }

    pub fn compile_isa_with_options<const WORD_SIZE: usize>(
        self,
        options: CompilerOptions,
    ) -> Program<F, AsmConfig<F, EF>> {
        let mut compiler = AsmCompiler::new(WORD_SIZE);
        compiler.build(self.operations);
        let asm_code = compiler.code();
        convert_program::<WORD_SIZE, F, EF, AsmConfig<F, EF>>(asm_code, options)
    }
}
