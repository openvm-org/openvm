use halo2_base::{
    gates::circuit::builder::BaseCircuitBuilder, halo2_proofs::halo2curves::bn256::Fr,
};
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
use snark_verifier_sdk::{
    evm::{gen_evm_proof_shplonk, gen_evm_verifier_sol_code},
    SHPLONK,
};

use super::{Halo2Params, Halo2Prover, StaticVerifierInput, StaticVerifierProvingKey};

/// EVM-compatible proof consisting of instances and raw proof bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawEvmProof {
    pub instances: Vec<Fr>,
    pub proof: Vec<u8>,
}

/// Compiled Solidity verifier contract for on-chain verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackEvmVerifier {
    pub sol_code: String,
    pub artifact: EvmVerifierByteCode,
}

/// Bytecode of a compiled EVM verifier contract.
#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct EvmVerifierByteCode {
    pub sol_compiler_version: String,
    pub sol_compiler_options: String,
    #[serde_as(as = "serde_with::hex::Hex")]
    pub bytecode: Vec<u8>,
}

impl StaticVerifierProvingKey {
    /// Generate a Solidity verifier contract for this circuit.
    pub fn generate_fallback_evm_verifier(&self, params: &Halo2Params) -> String {
        gen_evm_verifier_sol_code::<BaseCircuitBuilder<Fr>, SHPLONK>(
            params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
        )
    }

    /// Generate an EVM-compatible proof.
    pub fn prove_for_evm(
        &self,
        params: &Halo2Params,
        input: &StaticVerifierInput<'_>,
    ) -> RawEvmProof {
        let mut builder = BaseCircuitBuilder::prover(
            self.pinning.metadata.config_params.clone(),
            self.pinning.metadata.break_points.clone(),
        )
        .use_instance_columns(self.shape.instance_columns);

        let public_inputs = Halo2Prover::populate(&mut builder, input);

        let snark = gen_evm_proof_shplonk(
            params,
            &self.pinning.pk,
            builder,
            vec![public_inputs.clone()],
        );

        RawEvmProof {
            instances: public_inputs,
            proof: snark,
        }
    }
}

/// Verify an EVM proof using a deployed verifier contract.
///
/// Returns the gas used on success, or an error message on failure.
#[cfg(feature = "evm-verify")]
pub fn evm_verify(deployment_code: &[u8], proof: &RawEvmProof) -> Result<u64, String> {
    let calldata =
        snark_verifier_sdk::evm::encode_calldata(&[proof.instances.as_slice()], &proof.proof);
    snark_verifier_sdk::evm::evm_verify(deployment_code.to_vec(), calldata)
        .map_err(|e| format!("EVM verification failed: {e}"))
}
