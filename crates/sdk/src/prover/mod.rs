mod agg;
mod app;
#[cfg(feature = "evm-prove")]
mod halo2;
mod root;
mod stark;
pub mod vm;

pub use agg::*;
pub use app::*;
#[cfg(feature = "evm-prove")]
pub use evm::*;
#[cfg(feature = "evm-prove")]
pub use halo2::*;
pub use root::*;
pub use stark::*;

#[cfg(feature = "evm-prove")]
mod evm {
    use std::sync::Arc;

    use openvm_circuit::arch::{InsExecutorE1, VmConfig};
    use openvm_native_recursion::halo2::utils::Halo2ParamsReader;
    use openvm_stark_backend::config::Val;
    use openvm_stark_sdk::{engine::StarkFriEngine, openvm_stark_backend::Chip};

    use super::{Halo2Prover, StarkProver};
    use crate::{
        config::AggregationTreeConfig,
        keygen::{AggProvingKey, AppProvingKey},
        stdin::StdIn,
        types::EvmProof,
        NonRootCommittedExe, F, SC,
    };

    pub struct EvmHalo2Prover<VC, E: StarkFriEngine<SC>> {
        pub stark_prover: StarkProver<VC, E>,
        pub halo2_prover: Halo2Prover,
    }

    impl<VC, E: StarkFriEngine<SC>> EvmHalo2Prover<VC, E> {
        pub fn new(
            reader: &impl Halo2ParamsReader,
            app_pk: Arc<AppProvingKey<VC>>,
            app_committed_exe: Arc<NonRootCommittedExe>,
            agg_pk: AggProvingKey,
            agg_tree_config: AggregationTreeConfig,
        ) -> Self
        where
            VC: VmConfig<F>,
        {
            let AggProvingKey {
                agg_stark_pk,
                halo2_pk,
            } = agg_pk;
            let stark_prover =
                StarkProver::new(app_pk, app_committed_exe, agg_stark_pk, agg_tree_config);
            Self {
                stark_prover,
                halo2_prover: Halo2Prover::new(reader, halo2_pk),
            }
        }

        pub fn set_program_name(&mut self, program_name: impl AsRef<str>) -> &mut Self {
            self.stark_prover.set_program_name(program_name);
            self
        }

        pub fn generate_proof_for_evm(&self, input: StdIn) -> EvmProof
        where
            VC: VmConfig<F>,
            VC::Executor: Chip<SC> + InsExecutorE1<Val<SC>>,
            VC::Periphery: Chip<SC>,
        {
            let root_proof = self.stark_prover.generate_proof_for_outer_recursion(input);
            self.halo2_prover.prove_for_evm(&root_proof)
        }
    }
}
