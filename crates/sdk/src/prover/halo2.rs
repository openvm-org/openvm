use std::sync::Arc;

use openvm_continuations::RootSC;
use openvm_stark_backend::proof::Proof;
use openvm_static_verifier::{Halo2Params, Halo2ParamsReader};
use tracing::info_span;

use crate::{keygen::Halo2ProvingKey, types::EvmProof};

pub struct Halo2Prover {
    halo2_pk: Halo2ProvingKey,
    verifier_srs: Arc<Halo2Params>,
    wrapper_srs: Arc<Halo2Params>,
}

impl Halo2Prover {
    pub fn new(reader: &impl Halo2ParamsReader, halo2_pk: Halo2ProvingKey) -> Self {
        let verifier_k = halo2_pk.verifier.pinning.metadata.config_params.k;
        let wrapper_k = halo2_pk.wrapper.pinning.metadata.config_params.k;
        let verifier_srs = reader.read_params(verifier_k);
        let wrapper_srs = reader.read_params(wrapper_k);
        Self {
            halo2_pk,
            verifier_srs,
            wrapper_srs,
        }
    }

    pub fn prove_for_evm(&self, root_proof: &Proof<RootSC>) -> EvmProof {
        let snark = info_span!("prove", group = "halo2_outer").in_scope(|| {
            self.halo2_pk
                .verifier
                .prove_wrapped(&self.verifier_srs, root_proof)
        });
        info_span!("prove_for_evm", group = "halo2_wrapper").in_scope(|| {
            let raw = self
                .halo2_pk
                .wrapper
                .prove_for_evm(&self.wrapper_srs, snark);
            EvmProof::from(raw)
        })
    }
}
