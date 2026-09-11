use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    Executor, InterpretedInstance, PreflightCtx, StaticProgramError, VmExecutionConfig, VmExecutor,
};

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    /// Builds the interpreter preflight backend for differential tests.
    pub fn test_preflight_interpreter_instance(
        &self,
        exe: &VmExe,
    ) -> Result<InterpretedInstance<'_, PreflightCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_preflight", backend = "interpreter").entered();
        InterpretedInstance::new(&self.inventory, exe)
    }
}
