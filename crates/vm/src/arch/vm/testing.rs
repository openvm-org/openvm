use openvm_instructions::exe::VmExe;

use super::{
    Executor, InterpretedInstance, PreflightCtx, StaticProgramError, VmExecutionConfig, VmExecutor,
    VmField, VmFieldExecutionConfig, VmFieldExecutor,
};

macro_rules! impl_test_preflight_interpreter {
    (
        impl<$($generic:tt),+> $executor:ty,
        config: $config_trait:path
        $(, field: $field:ident)?
    ) => {
        impl<$($generic),+> $executor
        where
            $($field: VmField,)?
            VC: $config_trait,
            VC::Executor: Executor,
        {
            /// Builds the interpreter preflight backend for differential tests.
            pub fn test_preflight_interpreter_instance(
                &self,
                exe: &VmExe,
            ) -> Result<InterpretedInstance<'_, PreflightCtx>, StaticProgramError> {
                #[cfg(feature = "metrics")]
                let _compilation_span = tracing::info_span!(
                    "compile_preflight",
                    backend = "interpreter"
                )
                .entered();
                InterpretedInstance::new(&self.inventory, exe)
            }
        }
    };
}

impl_test_preflight_interpreter!(
    impl<VC> VmExecutor<VC>,
    config: VmExecutionConfig
);

impl_test_preflight_interpreter!(
    impl<F, VC> VmFieldExecutor<F, VC>,
    config: VmFieldExecutionConfig<F>,
    field: F
);
