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
        InsExecutorE1, InsExecutorE2, InstructionExecutor, VirtualMachineError, VmBuilder,
        VmExecutionConfig,
    };
    use openvm_native_circuit::NativeConfig;
    use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
    use openvm_stark_sdk::engine::StarkFriEngine;

    use super::{Halo2Prover, StarkProver};
    use crate::{
        config::AggregationTreeConfig,
        keygen::{AggProvingKey, AppProvingKey},
        stdin::StdIn,
        types::EvmProof,
        NonRootCommittedExe, F, SC,
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
        <VB::VmConfig as VmExecutionConfig<F>>::Executor: InsExecutorE1<F>
            + InsExecutorE2<F>
            + InstructionExecutor<F, <VB as VmBuilder<E>>::RecordArena>,
        NativeBuilder: VmBuilder<E, VmConfig = NativeConfig> + Clone,
        <NativeConfig as VmExecutionConfig<F>>::Executor:
            InstructionExecutor<F, <NativeBuilder as VmBuilder<E>>::RecordArena>,
    {
        pub fn new(
            reader: &impl Halo2ParamsReader,
            app_vm_builder: VB,
            native_builder: NativeBuilder,
            app_pk: Arc<AppProvingKey<VB::VmConfig>>,
            app_committed_exe: Arc<NonRootCommittedExe>,
            agg_pk: AggProvingKey,
            agg_tree_config: AggregationTreeConfig,
        ) -> Result<Self, VirtualMachineError> {
            let AggProvingKey {
                agg_stark_pk,
                halo2_pk,
            } = agg_pk;
            let stark_prover = StarkProver::new(
                app_vm_builder,
                native_builder,
                app_pk,
                app_committed_exe,
                agg_stark_pk,
                agg_tree_config,
            )?;
            Ok(Self {
                stark_prover,
                halo2_prover: Halo2Prover::new(reader, halo2_pk),
            })
        }

        pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
            self.stark_prover.set_program_name(program_name);
            self
        }

        pub fn generate_proof_for_evm(
            &mut self,
            input: StdIn,
        ) -> Result<EvmProof, VirtualMachineError> {
            let root_proof = self
                .stark_prover
                .generate_proof_for_outer_recursion(input)?;
            let evm_proof = self.halo2_prover.prove_for_evm(&root_proof);
            Ok(evm_proof)
        }
    }
}
