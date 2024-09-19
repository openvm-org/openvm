pub mod utils;
pub mod verifier;

pub mod testing_utils;
#[cfg(test)]
mod tests;

use std::{fmt::Debug, fs::File};

use afs_compiler::{
    constraints::{halo2::compiler::Halo2State, ConstraintCompiler},
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
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) -> BaseCircuitBuilder<Fr> {
        let mut state = Halo2State {
            builder,
            ..Default::default()
        };
        state.load_witness(witness);

        let backend = ConstraintCompiler::<C>::default();
        backend.constrain_halo2(&mut state, operations);

        state.builder
    }

    /// Executes the prover in testing mode with a circuit definition and witness.
    pub fn mock<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) {
        let builder = Self::builder(CircuitBuilderStage::Mock, k);
        let mut builder = Self::populate(builder, operations, witness);

        let public_instances = builder.instances();
        println!("Public instances: {:?}", public_instances);

        builder.calculate_params(Some(20));

        MockProver::run(k as u32, &builder, public_instances)
            .unwrap()
            .assert_satisfied();
    }

    /// Populates builder, tunes circuit, keygen
    pub fn keygen<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) -> Halo2ProvingPinning {
        let builder = Self::builder(CircuitBuilderStage::Keygen, k);
        let mut builder = Self::populate(builder, operations, witness);
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
        let pk = keygen_pk2(params.as_ref(), &builder, false).unwrap();
        let break_points = builder.break_points();

        let config_params = builder.config_params.clone();
        let num_pvs = builder
            .assigned_instances
            .iter()
            .map(|x| x.len())
            .collect_vec();

        let file = File::create("halo2_final.json").unwrap();
        serde_json::to_writer(file, &break_points).unwrap();
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
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) -> Snark {
        let k = config_params.k;
        let builder = Self::builder(CircuitBuilderStage::Prover, k)
            .use_params(config_params)
            .use_break_points(break_points);
        let builder = Self::populate(builder, operations, witness);
        #[cfg(feature = "bench-metrics")]
        {
            let stats = builder.statistics();
            let total_advices: usize = stats.gate.total_advice_per_phase.into_iter().sum();
            let total_lookups: usize = stats.total_lookup_advice_per_phase.into_iter().sum();
            let total_cell = total_advices + total_lookups + stats.gate.total_fixed;
            metrics::gauge!("halo2_total_cells").set(total_cell as f64);
        }

        let params = read_params(k as u32);

        #[cfg(feature = "bench-metrics")]
        let start = std::time::Instant::now();

        let snark = gen_snark_shplonk(&params, pk, builder, None::<&str>);

        #[cfg(feature = "bench-metrics")]
        metrics::gauge!("halo2_proof_time_ms").set(start.elapsed().as_millis() as f64);

        snark
    }

    pub fn full_prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) -> Snark {
        let Halo2ProvingPinning {
            pk,
            config_params,
            break_points,
            ..
        } = Self::keygen(k, operations.clone(), witness.clone());
        Self::prove(config_params, break_points, &pk, operations, witness)
    }
}
