use halo2_base::{
    gates::circuit::CircuitBuilderStage,
    halo2_proofs::plonk::{keygen_pk, keygen_vk},
};
use openvm_stark_sdk::{
    config::baby_bear_bn254_poseidon2::BabyBearBn254Poseidon2Config as RootConfig,
    openvm_stark_backend::proof::Proof,
};
use serde::{Deserialize, Serialize};

use crate::{
    circuit::StaticVerifierCircuit,
    config::StaticVerifierShape,
    prover::{Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning, StaticVerifierProof},
};

impl StaticVerifierCircuit {
    /// Run keygen to produce a [`Halo2ProvingPinning`].
    ///
    /// The `representative_proof` is used as a witness for keygen; any valid proof for this static
    /// circuit shape will do.
    pub fn keygen(
        &self,
        params: &Halo2Params,
        shape: &StaticVerifierShape,
        representative_proof: &Proof<RootConfig>,
    ) -> Halo2ProvingPinning {
        let mut builder = Self::builder(CircuitBuilderStage::Keygen, shape);
        let public_inputs = self.populate(&mut builder, representative_proof);

        let config_params = builder.calculate_params(Some(shape.minimum_rows));

        let vk = keygen_vk(params, &builder).expect("keygen_vk should succeed");
        let pk = keygen_pk(params, vk, &builder).expect("keygen_pk should succeed");
        let break_points = builder.break_points();

        Halo2ProvingPinning {
            pk,
            metadata: Halo2ProvingMetadata {
                config_params,
                break_points,
                num_pvs: vec![public_inputs.to_vec().len()],
            },
        }
    }
}

/// High-level proving key that owns a [`StaticVerifierCircuit`], [`Halo2ProvingPinning`], and
/// [`StaticVerifierShape`].
#[derive(Clone, Serialize, Deserialize)]
pub struct StaticVerifierProvingKey {
    pub circuit: StaticVerifierCircuit,
    pub pinning: Halo2ProvingPinning,
    pub shape: StaticVerifierShape,
}

impl StaticVerifierProvingKey {
    /// Run keygen and return a proving key that can be reused for multiple proofs.
    pub fn keygen(
        params: &Halo2Params,
        shape: StaticVerifierShape,
        circuit: StaticVerifierCircuit,
        representative_proof: &Proof<RootConfig>,
    ) -> Self {
        let pinning = circuit.keygen(params, &shape, representative_proof);
        Self {
            circuit,
            pinning,
            shape,
        }
    }

    /// Generate a proof using the stored pinning and shape.
    pub fn prove(&self, params: &Halo2Params, proof: &Proof<RootConfig>) -> StaticVerifierProof {
        self.circuit
            .prove(params, &self.pinning, &self.shape, proof)
    }

    /// Verify a proof against this proving key's verifying key.
    pub fn verify(&self, params: &Halo2Params, proof: &StaticVerifierProof) -> bool {
        StaticVerifierCircuit::verify(params, self.pinning.pk.get_vk(), proof)
    }
}

// --- EVM support (feature-gated) ---

#[cfg(feature = "evm-prove")]
use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder, halo2_proofs::halo2curves::bn256::Fr,
};
#[cfg(feature = "evm-prove")]
use snark_verifier_sdk::{
    evm::{gen_evm_proof_shplonk, gen_evm_verifier_sol_code},
    SHPLONK,
};

/// EVM-compatible proof consisting of instances and raw proof bytes.
#[cfg(feature = "evm-prove")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvmProof {
    pub instances: Vec<Fr>,
    pub proof: Vec<u8>,
}

#[cfg(feature = "evm-prove")]
impl StaticVerifierProvingKey {
    /// Generate a Solidity verifier contract for this circuit.
    pub fn generate_fallback_evm_verifier(&self, params: &Halo2Params) -> String {
        gen_evm_verifier_sol_code::<BaseCircuitBuilder<Fr>, SHPLONK>(
            params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
        )
    }

    /// Produce a [`Snark`] for consumption by the wrapper circuit.
    ///
    /// Unlike [`prove_for_evm_unwrapped`](Self::prove_for_evm_unwrapped), this
    /// returns a `Snark` (not a raw EVM proof), which should be fed into
    /// [`Halo2WrapperProvingKey::prove_for_evm`](crate::wrapper::Halo2WrapperProvingKey::prove_for_evm).
    pub fn prove_wrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> snark_verifier_sdk::Snark {
        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);

        let _public_inputs = self.circuit.populate(&mut builder, proof);

        snark_verifier_sdk::halo2::gen_snark_shplonk(
            params,
            &self.pinning.pk,
            builder,
            None::<&str>,
        )
    }

    /// Generate a dummy snark for wrapper keygen.
    pub fn generate_dummy_snark(
        &self,
        reader: &impl crate::wrapper::Halo2ParamsReader,
    ) -> snark_verifier_sdk::Snark {
        let k = self.pinning.metadata.config_params.k;
        let params = reader.read_params(k);
        snark_verifier_sdk::halo2::gen_dummy_snark_from_vk::<SHPLONK>(
            &params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
            None,
        )
    }

    /// Generate an EVM-compatible proof directly (one-step, no wrapper circuit).
    pub fn prove_for_evm_unwrapped(
        &self,
        params: &Halo2Params,
        proof: &Proof<RootConfig>,
    ) -> RawEvmProof {
        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);

        let public_inputs = self.circuit.populate(&mut builder, proof);
        let instances_vec = public_inputs.to_vec();

        let snark = gen_evm_proof_shplonk(
            params,
            &self.pinning.pk,
            builder,
            vec![instances_vec.clone()],
        );

        RawEvmProof {
            instances: instances_vec,
            proof: snark,
        }
    }
}

/// Verify an EVM proof using a deployed verifier contract.
///
/// Returns the gas used on success, or an error message on failure.
#[cfg(feature = "evm-verify")]
pub fn evm_verify(deployment_code: &[u8], proof: &RawEvmProof) -> Result<u64, String> {
    snark_verifier_sdk::evm::evm_verify(
        deployment_code.to_vec(),
        vec![proof.instances.clone()],
        proof.proof.clone(),
    )
    .map_err(|e| format!("EVM verification failed: {e}"))
}
