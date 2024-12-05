use itertools::Itertools;
use p3_util::log2_ceil_usize;
use snark_verifier_sdk::{
    evm::{evm_verify, gen_evm_proof_shplonk, gen_evm_verifier_shplonk},
    halo2::{
        aggregation::{AggregationCircuit, AggregationConfigParams, VerifierUniversality},
        gen_snark_shplonk,
    },
    snark_verifier::halo2_base::{
        gates::circuit::{
            CircuitBuilderStage,
            CircuitBuilderStage::{Keygen, Prover},
        },
        halo2_proofs::{halo2curves::bn256::Fr, plonk::keygen_pk2},
    },
    CircuitExt, Snark, SHPLONK,
};

use crate::halo2::{
    utils::{read_params, KZG_PARAMS_FOR_SVK},
    Halo2ProvingPinning,
};

#[derive(Debug, Clone)]
pub struct Halo2WrapperCircuit {
    pub pinning: Halo2ProvingPinning,
}

const MIN_ROWS: usize = 20;

impl Halo2WrapperCircuit {
    /// Auto select k to let Wrapper circuit only have 1 advice column.
    pub fn keygen_auto_tune(dummy_snark: Snark) -> Self {
        let k = Self::select_k(dummy_snark.clone());
        tracing::info!("Selected k: {}", k);
        Self::keygen(k, dummy_snark)
    }
    pub fn keygen(k: usize, dummy_snark: Snark) -> Self {
        let params = read_params(k as u32);
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let mut circuit = generate_wrapper_circuit_object(Keygen, k, dummy_snark);
        circuit.calculate_params(Some(MIN_ROWS));
        let config_params = circuit.builder.config_params.clone();
        tracing::info!(
            "Wrapper circuit num advice: {:?}",
            config_params.num_advice_per_phase
        );
        #[cfg(feature = "bench-metrics")]
        emit_wrapper_circuit_metrics(&circuit);
        let pk = keygen_pk2(params.as_ref(), &circuit, false).unwrap();
        let num_pvs = circuit.instances().iter().map(|x| x.len()).collect_vec();
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_keygen_time_ms").set(start.elapsed().as_millis() as f64);
        Self {
            pinning: Halo2ProvingPinning {
                pk,
                config_params,
                break_points: circuit.break_points(),
                num_pvs,
            },
        }
    }
    /// A helper function for testing to verify the proof of this circuit with evm verifier.
    // FIXME: the signature is not ideal. It should return an Error instead of panicking when the verification fails.
    pub fn evm_verify(
        evm_verifier_deployment_codes: Vec<u8>,
        evm_proof: Vec<u8>,
        pvs: Vec<Vec<Fr>>,
    ) {
        evm_verify(evm_verifier_deployment_codes, pvs, evm_proof);
    }
    /// Return deployment code for EVM verifier which can verify the snark of this circuit.
    pub fn generate_evm_verifier(&self) -> Vec<u8> {
        let params = read_params(self.pinning.config_params.k as u32);
        gen_evm_verifier_shplonk::<AggregationCircuit>(
            &params,
            self.pinning.pk.get_vk(),
            self.pinning.num_pvs.clone(),
            None,
        )
    }
    /// Return (EVM proof, public values)
    pub fn prove_for_evm(&self, snark_to_verify: Snark) -> (Vec<u8>, Vec<Vec<Fr>>) {
        let k = self.pinning.config_params.k;
        let params = read_params(k as u32);
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let prover_circuit = self.generate_circuit_object_for_proving(k, snark_to_verify);
        let pvs = prover_circuit.instances();
        let proof = gen_evm_proof_shplonk(&params, &self.pinning.pk, prover_circuit, pvs.clone());
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_proof_time_ms").set(start.elapsed().as_millis() as f64);

        (proof, pvs)
    }
    pub fn prove(&self, snark_to_verify: Snark) -> Snark {
        let k = self.pinning.config_params.k;
        let params = read_params(k as u32);
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let prover_circuit = self.generate_circuit_object_for_proving(k, snark_to_verify);
        let snark = gen_snark_shplonk(&params, &self.pinning.pk, prover_circuit, None::<String>);
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_proof_time_ms").set(start.elapsed().as_millis() as f64);
        snark
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
            self.pinning.num_pvs[0],
            // 12 is the number of public values for the accumulator
            snark_to_verify.instances[0].len() + 12
        );
        generate_wrapper_circuit_object(Prover, k, snark_to_verify)
            .use_params(self.pinning.config_params.clone().try_into().unwrap())
            .use_break_points(self.pinning.break_points.clone())
    }

    fn select_k(dummy_snark: Snark) -> usize {
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

#[cfg(test)]
mod tests {
    use snark_verifier_sdk::{
        halo2::gen_dummy_snark_from_vk,
        snark_verifier::{
            halo2_base::gates::circuit::builder::BaseCircuitBuilder, util::arithmetic::Field,
        },
    };

    use super::*;
    use crate::halo2::utils::gen_kzg_params;

    /// Return (dummy snark, real snark)
    fn snarks_dummy_circuit() -> (Snark, Snark) {
        let k = 10;
        let params = gen_kzg_params(k as u32);
        let mut builder = BaseCircuitBuilder::from_stage(Keygen)
            .use_k(k)
            .use_instance_columns(1);
        {
            let ctx = builder.main(0);
            let zero = ctx.load_constant(Field::ZERO);
            for _ in 0..2 * (1 << k) {
                ctx.load_witness(Field::ZERO);
            }
            builder.assigned_instances = vec![vec![zero]];
        }
        builder.calculate_params(Some(20));
        let config_params = builder.config_params.clone();
        let pk = keygen_pk2(&params, &builder, false).unwrap();
        let break_points = builder.break_points();
        let dummy_snark = gen_dummy_snark_from_vk::<SHPLONK>(&params, pk.get_vk(), vec![1], None);
        let mut builder = BaseCircuitBuilder::from_stage(Prover)
            .use_params(config_params)
            .use_break_points(break_points);
        {
            let ctx = builder.main(0);
            let zero = ctx.load_constant(Field::ZERO);
            ctx.load_witness(Field::ZERO);
            builder.assigned_instances = vec![vec![zero]];
        }
        let snark = gen_snark_shplonk(&params, &pk, builder, None::<&str>);
        (dummy_snark, snark)
    }
    #[test]
    fn test_select_k() {
        let (dummy_snark, _) = snarks_dummy_circuit();
        let wrapper_k = Halo2WrapperCircuit::select_k(dummy_snark);
        assert_eq!(wrapper_k, 22);
    }
}
