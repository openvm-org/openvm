use std::sync::Arc;

use halo2_base::{
    gates::circuit::CircuitBuilderStage,
    halo2_proofs::{
        halo2curves::bn256::G1Affine,
        plonk::keygen_pk2,
        poly::{
            commitment::{CommitmentScheme, Params},
            kzg::commitment::{KZGCommitmentScheme, ParamsKZG},
        },
    },
};
use itertools::Itertools;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
#[cfg(feature = "evm-prove")]
use snark_verifier_sdk::snark_verifier::{
    halo2_base::halo2_proofs::plonk::VerifyingKey, loader::evm::compile_solidity,
};
use snark_verifier_sdk::{
    halo2::aggregation::{AggregationCircuit, AggregationConfigParams, VerifierUniversality},
    CircuitExt, Snark, SHPLONK,
};

use crate::{
    keygen::RawEvmProof,
    prover::{Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning},
};

// ---- KZG params for SVK (ported from openvm-main utils.rs) ----

static SVK: Lazy<G1Affine> = Lazy::new(|| {
    serde_json::from_str("\"0100000000000000000000000000000000000000000000000000000000000000\"")
        .unwrap()
});

/// Hacking because of bad interface. This is to construct a fake KZG params to pass
/// Svk (which only requires ParamsKZG.g[0]) to AggregationCircuit.
static FAKE_KZG_PARAMS: Lazy<Halo2Params> = Lazy::new(|| KZGCommitmentScheme::new_params(1));

pub static KZG_PARAMS_FOR_SVK: Lazy<Halo2Params> = Lazy::new(|| {
    if std::env::var("RANDOM_SRS").is_ok() {
        // For testing: use a random SRS
        use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let mut params = ParamsKZG::setup(23, &mut rng);
        params.downsize(1);
        params
    } else {
        build_kzg_params_for_svk(*SVK)
    }
});

fn build_kzg_params_for_svk(g: G1Affine) -> Halo2Params {
    FAKE_KZG_PARAMS.from_parts(
        1,
        vec![g],
        Some(vec![g]),
        Default::default(),
        Default::default(),
    )
}

// ---- Halo2ParamsReader trait ----

/// Trait for reading Halo2 KZG parameters by degree `k`.
pub trait Halo2ParamsReader {
    fn read_params(&self, k: usize) -> Arc<Halo2Params>;
}

// ---- Wrapper types ----

/// `FallbackEvmVerifier` is for the raw verifier contract outputted by
/// `snark-verifier` for on-chain verification
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halo2WrapperProvingKey {
    pub pinning: Halo2ProvingPinning,
}

const MIN_ROWS: usize = 20;

impl Halo2WrapperProvingKey {
    /// Auto select k to let Wrapper circuit only have 1 advice column.
    pub fn keygen_auto_tune(reader: &impl Halo2ParamsReader, dummy_snark: Snark) -> Self {
        let k = Self::select_k(dummy_snark.clone());
        tracing::info!("Selected wrapper k: {k}");
        let params = reader.read_params(k);
        Self::keygen(&params, dummy_snark)
    }

    pub fn keygen(params: &Halo2Params, dummy_snark: Snark) -> Self {
        let k = params.k();
        let mut circuit =
            generate_wrapper_circuit_object(CircuitBuilderStage::Keygen, k as usize, dummy_snark);
        circuit.calculate_params(Some(MIN_ROWS));
        let config_params = circuit.builder.config_params.clone();
        tracing::info!(
            "Wrapper circuit num advice: {:?}",
            config_params.num_advice_per_phase
        );
        let pk = keygen_pk2(params, &circuit, false).unwrap();
        let num_pvs = circuit.instances().iter().map(|x| x.len()).collect_vec();
        Self {
            pinning: Halo2ProvingPinning {
                pk,
                metadata: Halo2ProvingMetadata {
                    config_params,
                    break_points: circuit.break_points(),
                    num_pvs,
                },
            },
        }
    }

    #[cfg(feature = "evm-verify")]
    /// A helper function for testing to verify the proof of this circuit with evm verifier.
    pub fn evm_verify(
        evm_verifier: &FallbackEvmVerifier,
        evm_proof: &RawEvmProof,
    ) -> Result<u64, String> {
        snark_verifier_sdk::evm::evm_verify(
            evm_verifier.artifact.bytecode.clone(),
            vec![evm_proof.instances.clone()],
            evm_proof.proof.clone(),
        )
    }

    #[cfg(feature = "evm-prove")]
    /// Return deployment code for EVM verifier which can verify the snark of this circuit.
    pub fn generate_fallback_evm_verifier(&self, params: &Halo2Params) -> FallbackEvmVerifier {
        assert_eq!(
            self.pinning.metadata.config_params.k as u32,
            params.k(),
            "Provided params don't match circuit config"
        );
        gen_evm_verifier(
            params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
        )
    }

    #[cfg(feature = "evm-prove")]
    pub fn prove_for_evm(&self, params: &Halo2Params, snark_to_verify: Snark) -> RawEvmProof {
        let k = self.pinning.metadata.config_params.k;
        let prover_circuit = self.generate_circuit_object_for_proving(k, snark_to_verify);
        let mut pvs = prover_circuit.instances();
        assert_eq!(pvs.len(), 1);
        let proof = snark_verifier_sdk::evm::gen_evm_proof_shplonk(
            params,
            &self.pinning.pk,
            prover_circuit,
            pvs.clone(),
        );

        RawEvmProof {
            instances: pvs.pop().unwrap(),
            proof,
        }
    }

    #[cfg(feature = "evm-prove")]
    fn generate_circuit_object_for_proving(
        &self,
        k: usize,
        snark_to_verify: Snark,
    ) -> AggregationCircuit {
        assert_eq!(
            snark_to_verify.instances.len(),
            1,
            "Snark should only have 1 instance column"
        );
        assert_eq!(
            self.pinning.metadata.num_pvs[0],
            snark_to_verify.instances[0].len() + 12,
        );
        generate_wrapper_circuit_object(CircuitBuilderStage::Prover, k, snark_to_verify)
            .use_params(
                self.pinning
                    .metadata
                    .config_params
                    .clone()
                    .try_into()
                    .unwrap(),
            )
            .use_break_points(self.pinning.metadata.break_points.clone())
    }

    pub(crate) fn select_k(dummy_snark: Snark) -> usize {
        let mut k = 20;
        let mut first_run = true;
        loop {
            let mut circuit = generate_wrapper_circuit_object(
                CircuitBuilderStage::Keygen,
                k,
                dummy_snark.clone(),
            );
            circuit.calculate_params(Some(MIN_ROWS));
            assert_eq!(
                circuit.builder.config_params.num_advice_per_phase.len(),
                1,
                "Snark has multiple phases"
            );
            if circuit.builder.config_params.num_advice_per_phase[0] == 1 {
                circuit.builder.clear();
                break;
            }
            if first_run {
                k = log2_ceil_usize(
                    circuit.builder.statistics().gate.total_advice_per_phase[0] + MIN_ROWS,
                );
            } else {
                k += 1;
            }
            first_run = false;
            // Prevent drop warnings
            circuit.builder.clear();
        }
        k
    }
}

fn generate_wrapper_circuit_object(
    stage: CircuitBuilderStage,
    k: usize,
    snark: Snark,
) -> AggregationCircuit {
    let config_params = AggregationConfigParams {
        degree: k as u32,
        lookup_bits: k - 1,
        ..Default::default()
    };
    let mut circuit = AggregationCircuit::new::<SHPLONK>(
        stage,
        config_params,
        &KZG_PARAMS_FOR_SVK,
        [snark],
        VerifierUniversality::None,
    );
    circuit.expose_previous_instances(false);
    circuit
}

#[cfg(feature = "evm-prove")]
fn gen_evm_verifier(
    params: &Halo2Params,
    vk: &VerifyingKey<G1Affine>,
    num_instance: Vec<usize>,
) -> FallbackEvmVerifier {
    let sol_code = snark_verifier_sdk::evm::gen_evm_verifier_sol_code::<AggregationCircuit, SHPLONK>(
        params,
        vk,
        num_instance,
    );
    let byte_code = compile_solidity(&sol_code);
    FallbackEvmVerifier {
        sol_code,
        artifact: EvmVerifierByteCode {
            sol_compiler_version: "0.8.19".to_string(),
            sol_compiler_options: "".to_string(),
            bytecode: byte_code,
        },
    }
}

/// Compute ceil(log2(n)) for n > 0.
fn log2_ceil_usize(n: usize) -> usize {
    assert!(n > 0);
    (usize::BITS - (n - 1).leading_zeros()) as usize
}
