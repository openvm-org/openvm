use ax_stark_backend::prover::types::Proof;
use ax_stark_sdk::config::{baby_bear_poseidon2_root::BabyBearPoseidon2RootConfig, FriParameters};
use axvm_native_compiler::ir::Witness;
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
    pub fn prove(&self, params: &Halo2Params, witness: Witness<OuterConfig>) -> Snark {
        Halo2Prover::prove(
            params,
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
            &self.pinning.pk,
            self.dsl_ops.clone(),
            witness,
        )
    }
    // TODO: Add verify method

    /// Generate a dummy snark for wrapper keygen.
    pub fn generate_dummy_snark(&self, reader: &impl Halo2ParamsReader) -> Snark {
        self.pinning.generate_dummy_snark(reader)
    }
}
