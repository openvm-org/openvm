pub mod utils;
pub mod verifier;

pub mod testing_utils;
#[cfg(test)]
mod tests;

use std::fmt::Debug;

use axvm_native_compiler::{
    constraints::halo2::compiler::{Halo2ConstraintCompiler, Halo2State},
    ir::{Config, DslIr, TracedVec, Witness},
};
use itertools::Itertools;
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::extension::BinomialExtensionField;
use snark_verifier_sdk::{
    halo2::gen_snark_shplonk,
    snark_verifier::halo2_base::{
        gates::{
            circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
            flex_gate::MultiPhaseThreadBreakPoints,
        },
        halo2_proofs::{
            dev::MockProver,
            halo2curves::bn256::{Fr, G1Affine},
            plonk::{keygen_pk2, ProvingKey},
        },
    },
    CircuitExt, Snark,
};

use crate::halo2::utils::read_params;

/// A prover that can generate proofs with the Halo2
#[derive(Debug, Clone)]
pub struct Halo2Prover;

#[derive(Debug, Clone)]
pub struct DslOperations<C: Config> {
    pub operations: TracedVec<DslIr<C>>,
    pub num_public_values: usize,
}

/// Necessary metadata to prove a Halo2 circuit
#[derive(Debug, Clone)]
pub struct Halo2ProvingPinning {
    pub pk: ProvingKey<G1Affine>,
    pub config_params: BaseCircuitParams,
    pub break_points: MultiPhaseThreadBreakPoints,
    /// Number of public values per column in order.
    pub num_pvs: Vec<usize>,
}

impl Halo2Prover {
    pub fn builder(stage: CircuitBuilderStage, k: usize) -> BaseCircuitBuilder<Fr> {
        BaseCircuitBuilder::from_stage(stage)
            .use_k(k)
            .use_lookup_bits(k - 1)
            .use_instance_columns(1)
    }

    pub fn populate<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        builder: BaseCircuitBuilder<Fr>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
        #[allow(unused_variables)] collect_metrics: bool,
    ) -> BaseCircuitBuilder<Fr> {
        let mut state = Halo2State {
            builder,
            ..Default::default()
        };
        state.load_witness(witness);

        let backend = Halo2ConstraintCompiler::<C>::new(dsl_operations.num_public_values);
        #[cfg(feature = "bench-metrics")]
        let backend = if collect_metrics {
            backend.with_collect_metrics()
        } else {
            backend
        };
        backend.constrain_halo2(&mut state, dsl_operations.operations);

        state.builder
    }

    /// Executes the prover in testing mode with a circuit definition and witness.
    ///
    /// Returns the public instances.
    pub fn mock<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Vec<Vec<Fr>> {
        let builder = Self::builder(CircuitBuilderStage::Mock, k);
        let mut builder = Self::populate(builder, dsl_operations, witness, true);

        let public_instances = builder.instances();
        println!("Public instances: {:?}", public_instances);

        builder.calculate_params(Some(20));

        MockProver::run(k as u32, &builder, public_instances.clone())
            .unwrap()
            .assert_satisfied();
        public_instances
    }

    /// Populates builder, tunes circuit, keygen
    pub fn keygen<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Halo2ProvingPinning {
        let builder = Self::builder(CircuitBuilderStage::Keygen, k);
        let mut builder = Self::populate(builder, dsl_operations, witness, true);
        builder.calculate_params(Some(20));

        let params = read_params(k as u32);
        // let break_points;
        // // if pk already exists, read break points from file
        // let pk = if Path::new("halo2_final.pk").exists() {
        //     let file = File::open("halo2_final.json").unwrap();
        //     break_points = serde_json::from_reader(file).unwrap();
        //     gen_pk(&params, &builder, Some(Path::new("halo2_final.pk")))
        // } else {
        //
        //     pk
        // };
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let pk = keygen_pk2(params.as_ref(), &builder, false).unwrap();
        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_keygen_time_ms").set(start.elapsed().as_millis() as f64);
        let break_points = builder.break_points();

        let config_params = builder.config_params.clone();
        let num_pvs = builder
            .assigned_instances
            .iter()
            .map(|x| x.len())
            .collect_vec();

        // let file = File::create("halo2_final.json").unwrap();
        // serde_json::to_writer(file, &break_points).unwrap();
        Halo2ProvingPinning {
            pk,
            config_params,
            break_points,
            num_pvs,
        }
    }

    pub fn prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        config_params: BaseCircuitParams,
        break_points: MultiPhaseThreadBreakPoints,
        pk: &ProvingKey<G1Affine>,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Snark {
        let k = config_params.k;
        let params = read_params(k as u32);
        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();
        let builder = Self::builder(CircuitBuilderStage::Prover, k)
            .use_params(config_params)
            .use_break_points(break_points);
        let builder = Self::populate(builder, dsl_operations, witness, false);
        #[cfg(feature = "bench-metrics")]
        {
            let stats = builder.statistics();
            let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
            let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
            let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
            metrics::gauge!("halo2_total_cells").set(total_cell as f64);
        }
        let snark = gen_snark_shplonk(&params, pk, builder, None::<&str>);

        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_proof_time_ms").set(start.elapsed().as_millis() as f64);

        snark
    }

    pub fn full_prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        dsl_operations: DslOperations<C>,
        witness: Witness<C>,
    ) -> Snark {
        let Halo2ProvingPinning {
            pk,
            config_params,
            break_points,
            ..
        } = Self::keygen(k, dsl_operations.clone(), witness.clone());
        Self::prove(config_params, break_points, &pk, dsl_operations, witness)
    }
}
