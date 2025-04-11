use std::sync::Arc;

use openvm_circuit::arch::VmConfig;
use openvm_stark_sdk::{engine::StarkFriEngine, openvm_stark_backend::Chip};

use crate::{
    config::AggregationTreeConfig, keygen::AppProvingKey, stdin::StdIn, NonRootCommittedExe, F, SC,
};

mod agg;
pub use agg::*;
mod app;
pub use app::*;
use openvm_native_recursion::halo2::utils::Halo2ParamsReader;

mod halo2;
#[allow(unused_imports)]
pub use halo2::*;
mod root;
pub use root::*;
mod stark;
pub mod vm;

#[allow(unused_imports)]
pub use stark::*;

use crate::{keygen::AggProvingKey, prover::halo2::Halo2Prover, types::EvmProof};

pub struct ContinuationProver<VC, E: StarkFriEngine<SC>> {
    pub stark_prover: StarkProver<VC, E>,
    pub halo2_prover: Halo2Prover,
}

impl<VC, E: StarkFriEngine<SC>> ContinuationProver<VC, E> {
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
        VC::Executor: Chip<SC>,
        VC::Periphery: Chip<SC>,
    {
        let root_proof = self.stark_prover.generate_proof_for_outer_recursion(input);
        self.halo2_prover.prove_for_evm(&root_proof)
    }
}
