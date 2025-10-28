mod agg;
mod app;
#[cfg(feature = "evm-prove")]
mod halo2;
#[cfg(feature = "evm-prove")]
mod root;
mod stark;
pub mod vm;

pub use agg::*;
pub use app::*;
#[cfg(feature = "evm-prove")]
pub use evm::*;
#[cfg(feature = "evm-prove")]
pub use halo2::*;
#[cfg(feature = "evm-prove")]
pub use root::*;
pub use stark::*;

#[cfg(feature = "evm-prove")]
mod evm {
    use std::sync::Arc;

    use openvm_circuit::arch::{
        instructions::exe::VmExe, Executor, MeteredExecutor, PreflightExecutor,
        VirtualMachineError, VmBuilder, VmExecutionConfig,
    };
    use openvm_native_circuit::NativeConfig;
    use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
    use openvm_stark_sdk::engine::StarkFriEngine;

    use super::{Halo2Prover, StarkProver};
    use crate::{
        config::AggregationTreeConfig,
        keygen::{AggProvingKey, AppProvingKey, Halo2ProvingKey},
        stdin::StdIn,
        types::EvmProof,
        F, SC,
    };

    pub struct EvmHalo2Prover<E, VB, NativeBuilder>
    where
        E: StarkFriEngine<SC = SC>,
        VB: VmBuilder<E>,
        NativeBuilder: VmBuilder<E, VmConfig = NativeConfig>,
    {
        pub stark_prover: StarkProver<E, VB, NativeBuilder>,
        pub halo2_prover: Halo2Prover,
    }

    impl<E, VB, NativeBuilder> EvmHalo2Prover<E, VB, NativeBuilder>
    where
        E: StarkFriEngine<SC = SC>,
        VB: VmBuilder<E>,
        <VB::VmConfig as VmExecutionConfig<F>>::Executor: Executor<F>
            + MeteredExecutor<F>
            + PreflightExecutor<F, <VB as VmBuilder<E>>::RecordArena>,
        NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone,
        <NativeConfig as VmExecutionConfig<F>>::Executor:
            PreflightExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
    {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            reader: &impl Halo2ParamsReader,
            app_vm_builder: VB,
            native_builder: NativeBuilder,
            app_pk: &AppProvingKey<VB::VmConfig>,
            app_exe: Arc<VmExe<F>>,
            agg_pk: &AggProvingKey,
            halo2_pk: Halo2ProvingKey,
            agg_tree_config: AggregationTreeConfig,
        ) -> Result<Self, VirtualMachineError> {
            let stark_prover = StarkProver::new(
                app_vm_builder,
                native_builder,
                app_pk,
                app_exe,
                agg_pk,
                agg_tree_config,
            )?;
            Ok(Self {
                stark_prover,
                halo2_prover: Halo2Prover::new(reader, halo2_pk),
            })
        }

        pub fn with_program_name(mut self, program_name: impl AsRef<str>) -> Self {
            self.set_program_name(program_name);
            self
        }
        pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
            self.stark_prover.set_program_name(program_name);
            self
        }

        pub fn prove_evm(&mut self, input: StdIn) -> Result<EvmProof, VirtualMachineError> {
            let root_proof = self
                .stark_prover
                .generate_proof_for_outer_recursion(input)?;
            let evm_proof = self.halo2_prover.prove_for_evm(&root_proof);
            Ok(evm_proof)
        }
    }
}
