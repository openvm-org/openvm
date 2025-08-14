use openvm_native_compiler::ir::Witness;
use openvm_stark_backend::proof::Proof;
use openvm_stark_sdk::config::{
    baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig, FriParameters,
};
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::Snark;

use crate::{
    config::outer::OuterConfig,
    halo2::{
        utils::Halo2ParamsReader, DslOperations, Halo2Params, Halo2Prover, Halo2ProvingPinning,
    },
    stark::outer::build_circuit_verify_operations,
    types::MultiStarkVerificationAdvice,
    witness::Witnessable,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halo2VerifierProvingKey {
    pub pinning: Halo2ProvingPinning,
    pub dsl_ops: DslOperations<OuterConfig>,
}

/// Generate a Halo2 verifier circuit for a given stark.
pub fn generate_halo2_verifier_proving_key(
    params: &Halo2Params,
    advice: MultiStarkVerificationAdvice<OuterConfig>,
    fri_params: &FriParameters,
    proof: &Proof<BabyBearPoseidon2RootConfig>,
) -> Halo2VerifierProvingKey {
    let mut witness = Witness::default();
    proof.write(&mut witness);
    let dsl_operations = build_circuit_verify_operations(advice, fri_params, proof);
    Halo2VerifierProvingKey {
        pinning: Halo2Prover::keygen(params, dsl_operations.clone(), witness),
        dsl_ops: dsl_operations,
    }
}

impl Halo2VerifierProvingKey {
    pub fn prove(
        &self,
        params: &Halo2Params,
        witness: Witness<OuterConfig>,
        profiling: bool,
    ) -> Snark {
        Halo2Prover::prove(
            params,
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
            &self.pinning.pk,
            self.dsl_ops.clone(),
            witness,
            profiling,
        )
    }
    // TODO: Add verify method

    /// Generate a dummy snark for wrapper keygen.
    pub fn generate_dummy_snark(&self, reader: &impl Halo2ParamsReader) -> Snark {
        self.pinning.generate_dummy_snark(reader)
    }
}

// SAFETY: the reason these aren't auto-implemented is because DslOperations contains TracedVec
// which has backtrace. This is only used for debugging purposes, and the rest of the proving key is
// Send and Sync.
unsafe impl Send for Halo2VerifierProvingKey {}
unsafe impl Sync for Halo2VerifierProvingKey {}
