use itertools::Itertools;
use openvm_stark_backend::p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;
#[cfg(feature = "evm-prove")]
use snark_verifier_sdk::snark_verifier::{
    halo2_base::halo2_proofs::{halo2curves::bn256::G1Affine, plonk::VerifyingKey},
    loader::evm::compile_solidity,
};
use snark_verifier_sdk::{
    halo2::aggregation::{AggregationCircuit, AggregationConfigParams, VerifierUniversality},
    snark_verifier::halo2_base::{
        gates::circuit::CircuitBuilderStage,
        halo2_proofs::{plonk::keygen_pk2, poly::commitment::Params},
    },
    CircuitExt, Snark, SHPLONK,
};

#[cfg(feature = "evm-prove")]
use crate::halo2::RawEvmProof;
use crate::halo2::{
    utils::{Halo2ParamsReader, KZG_PARAMS_FOR_SVK},
    Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning,
};

/// `FallbackEvmVerifier` is for the raw verifier contract outputted by
/// `snark-verifier`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackEvmVerifier {
    pub sol_code: String,
    pub artifact: EvmVerifierByteCode,
}

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
        tracing::info!("Selected k: {}", k);
        let params = reader.read_params(k);
        Self::keygen(&params, dummy_snark)
    }
    pub fn keygen(params: &Halo2Params, dummy_snark: Snark) -> Self {
        let k = params.k();
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        let mut circuit =
            generate_wrapper_circuit_object(CircuitBuilderStage::Keygen, k as usize, dummy_snark);
        circuit.calculate_params(Some(MIN_ROWS));
        let config_params = circuit.builder.config_params.clone();
        tracing::info!(
            "Wrapper circuit num advice: {:?}",
            config_params.num_advice_per_phase
        );
        #[cfg(feature = "metrics")]
        emit_wrapper_circuit_metrics(&circuit);
        let pk = keygen_pk2(params, &circuit, false).unwrap();
        let num_pvs = circuit.instances().iter().map(|x| x.len()).collect_vec();
        #[cfg(feature = "metrics")]
        metrics::gauge!("halo2_keygen_time_ms").set(start.elapsed().as_millis() as f64);
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
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
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
        #[cfg(feature = "metrics")]
        metrics::gauge!("total_proof_time_ms").set(start.elapsed().as_millis() as f64);

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

#[cfg(feature = "metrics")]
fn emit_wrapper_circuit_metrics(agg_circuit: &AggregationCircuit) {
    let stats = agg_circuit.builder.statistics();
    let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
    let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
    let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
    metrics::gauge!("halo2_total_cells").set(total_cell as f64);
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
