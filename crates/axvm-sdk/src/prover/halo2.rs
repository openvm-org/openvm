use std::sync::Arc;

use ax_stark_sdk::ax_stark_backend::prover::types::Proof;
use axvm_native_compiler::prelude::Witness;
use axvm_native_recursion::{
    halo2::{utils::read_params, EvmProof, Halo2Params},
    witness::Witnessable,
};
use tracing::info_span;

use crate::{keygen::Halo2ProvingKey, RootSC};
pub struct Halo2Prover {
    halo2_pk: Halo2ProvingKey,
    verifier_srs: Arc<Halo2Params>,
    wrapper_srs: Arc<Halo2Params>,
}

impl Halo2Prover {
    pub fn new(halo2_pk: Halo2ProvingKey) -> Self {
        let verifier_k = halo2_pk.verifier.pinning.metadata.config_params.k;
        let wrapper_k = halo2_pk.wrapper.pinning.metadata.config_params.k;
        let verifier_srs = read_params(verifier_k as u32);
        let wrapper_srs = if verifier_k != wrapper_k {
            read_params(wrapper_k as u32)
        } else {
            verifier_srs.clone()
        };
        Self {
            halo2_pk,
            verifier_srs,
            wrapper_srs,
        }
    }
    pub fn prove_for_evm(&self, root_proof: &Proof<RootSC>) -> EvmProof {
        let mut witness = Witness::default();
        root_proof.write(&mut witness);
        let snark = info_span!("halo2 verifier", group = "halo2_verifier").in_scope(|| {
            self.halo2_pk
                .verifier
                .prove_with_loaded_params(&self.verifier_srs, witness)
        });
        info_span!("halo2 wrapper", group = "halo2_wrapper").in_scope(|| {
            self.halo2_pk
                .wrapper
                .prove_for_evm_with_loaded_params(&self.wrapper_srs, snark)
        })
    }
}
