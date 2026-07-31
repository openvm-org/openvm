use std::sync::{Arc, Mutex};

use openvm_continuations::RootSC;
use openvm_stark_backend::proof::Proof;
use openvm_static_verifier::{
    tracegen::graph_executor::GraphExecutorState, Halo2Params, Halo2ParamsReader,
};
use tracing::{info, info_span};

use crate::{
    keygen::Halo2ProvingKey,
    types::{EvmProof, EvmProofConversionError},
};

#[derive(Clone)]
pub struct Halo2Prover {
    halo2_pk: Halo2ProvingKey,
    verifier_srs: Arc<Halo2Params>,
    wrapper_srs: Arc<Halo2Params>,
    verifier_graph_state: Arc<Mutex<GraphExecutorState>>,
}

impl Halo2Prover {
    pub fn new(reader: &impl Halo2ParamsReader, halo2_pk: Halo2ProvingKey) -> Self {
        let verifier_k = halo2_pk.verifier.pinning.metadata.config_params.k;
        let wrapper_k = halo2_pk.wrapper.pinning.metadata.config_params.k;
        info!(verifier_k, wrapper_k, "Halo2Prover initialized");
        let verifier_srs = reader.read_params(verifier_k);
        let wrapper_srs = reader.read_params(wrapper_k);
        let verifier_graph_state = Arc::new(Mutex::new(GraphExecutorState::new(
            &halo2_pk.verifier.graph_program,
        )));
        Self {
            halo2_pk,
            verifier_srs,
            wrapper_srs,
            verifier_graph_state,
        }
    }

    pub fn prove_for_evm(
        &self,
        root_proof: &Proof<RootSC>,
    ) -> Result<EvmProof, EvmProofConversionError> {
        #[cfg(feature = "metrics")]
        {
            let verifier_k = self.halo2_pk.verifier.pinning.metadata.config_params.k;
            let wrapper_k = self.halo2_pk.wrapper.pinning.metadata.config_params.k;
            metrics::gauge!("halo2_verifier_k", "group" => "halo2_outer").set(verifier_k as f64);
            metrics::gauge!("halo2_wrapper_k", "group" => "halo2_wrapper").set(wrapper_k as f64);
        }
        let snark = info_span!("total_proof", group = "halo2_outer").in_scope(|| {
            let mut state = self.verifier_graph_state.lock().unwrap();
            self.halo2_pk
                .verifier
                .prove_wrapped(&self.verifier_srs, root_proof, &mut state)
        });
        info_span!("total_proof", group = "halo2_wrapper").in_scope(|| {
            let raw = self
                .halo2_pk
                .wrapper
                .prove_for_evm(&self.wrapper_srs, snark);
            EvmProof::try_from(raw)
        })
    }

    pub fn pk(&self) -> Halo2ProvingKey {
        self.halo2_pk.clone()
    }
}
