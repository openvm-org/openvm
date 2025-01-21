use itertools::Itertools;
use openvm_stark_backend::p3_util::log2_ceil_usize;
use serde::{Deserialize, Serialize};
use snark_verifier_sdk::{
    evm::{evm_verify, gen_evm_proof_shplonk, gen_evm_verifier_shplonk},
    halo2::aggregation::{AggregationCircuit, AggregationConfigParams, VerifierUniversality},
    snark_verifier::halo2_base::{
        gates::circuit::{
            CircuitBuilderStage,
            CircuitBuilderStage::{Keygen, Prover},
        },
        halo2_proofs::{plonk::keygen_pk2, poly::commitment::Params},
    },
    CircuitExt, Snark, SHPLONK,
};

use crate::halo2::{
    utils::{Halo2ParamsReader, KZG_PARAMS_FOR_SVK},
    EvmProof, Halo2Params, Halo2ProvingMetadata, Halo2ProvingPinning,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvmVerifier(pub Vec<u8>);

impl From<Vec<u8>> for EvmVerifier {
    fn from(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }
}

impl From<EvmVerifier> for Vec<u8> {
    fn from(verifier: EvmVerifier) -> Self {
        verifier.0
    }
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
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let mut circuit = generate_wrapper_circuit_object(Keygen, k as usize, dummy_snark);
        circuit.calculate_params(Some(MIN_ROWS));
        let config_params = circuit.builder.config_params.clone();
        tracing::info!(
            "Wrapper circuit num advice: {:?}",
            config_params.num_advice_per_phase
        );
        #[cfg(feature = "bench-metrics")]
        emit_wrapper_circuit_metrics(&circuit);
        let pk = keygen_pk2(params, &circuit, false).unwrap();
        let num_pvs = circuit.instances().iter().map(|x| x.len()).collect_vec();
        #[cfg(feature = "bench-metrics")]
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
    /// A helper function for testing to verify the proof of this circuit with evm verifier.
    // FIXME: the signature is not ideal. It should return an Error instead of panicking when the verification fails.
    pub fn evm_verify(evm_verifier: &EvmVerifier, evm_proof: &EvmProof) {
        evm_verify(
            evm_verifier.0.clone(),
            evm_proof.instances.clone(),
            evm_proof.proof.clone(),
        );
    }
    /// Return deployment code for EVM verifier which can verify the snark of this circuit.
    pub fn generate_evm_verifier(&self, params: &Halo2Params) -> EvmVerifier {
        assert_eq!(
            self.pinning.metadata.config_params.k as u32,
            params.k(),
            "Provided params don't match circuit config"
        );
        EvmVerifier(gen_evm_verifier_shplonk::<AggregationCircuit>(
            params,
            self.pinning.pk.get_vk(),
            self.pinning.metadata.num_pvs.clone(),
            None,
        ))
    }
    pub fn prove_for_evm(&self, params: &Halo2Params, snark_to_verify: Snark) -> EvmProof {
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let k = self.pinning.metadata.config_params.k;
        let prover_circuit = self.generate_circuit_object_for_proving(k, snark_to_verify);
        let pvs = prover_circuit.instances();
        let proof = gen_evm_proof_shplonk(params, &self.pinning.pk, prover_circuit, pvs.clone());
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("total_proof_time_ms").set(start.elapsed().as_millis() as f64);

        EvmProof {
            instances: pvs,
            proof,
        }
    }
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
        generate_wrapper_circuit_object(Prover, k, snark_to_verify)
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
            let mut circuit = generate_wrapper_circuit_object(Keygen, k, dummy_snark.clone());
            circuit.calculate_params(Some(MIN_ROWS));
            assert_eq!(
                circuit.builder.config_params.num_advice_per_phase.len(),
                1,
                "Snark has multiple phases"
            );
            if circuit.builder.config_params.num_advice_per_phase[0] == 1 {
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

#[cfg(feature = "bench-metrics")]
fn emit_wrapper_circuit_metrics(agg_circuit: &AggregationCircuit) {
    let stats = agg_circuit.builder.statistics();
    let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
    let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
    let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
    metrics::gauge!("halo2_total_cells").set(total_cell as f64);
}
