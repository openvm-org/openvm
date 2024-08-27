use std::{fmt::Debug, fs::File, path::Path};

use afs_compiler::{
    constraints::{halo2::compiler::Halo2State, ConstraintCompiler},
    ir::{Config, DslIr, TracedVec, Witness},
};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::extension::BinomialExtensionField;
use snark_verifier_sdk::{
    gen_pk,
    halo2::gen_snark_shplonk,
    snark_verifier::halo2_base::{
        gates::{
            circuit::{builder::BaseCircuitBuilder, BaseCircuitParams, CircuitBuilderStage},
            flex_gate::MultiPhaseThreadBreakPoints,
        },
        halo2_proofs::{
            dev::MockProver,
            halo2curves::bn256::{Fr, G1Affine},
            plonk::ProvingKey,
        },
        utils::fs::read_params,
    },
    CircuitExt, Snark,
};

/// A prover that can generate proofs with the Halo2
#[derive(Debug, Clone)]
pub struct Halo2Prover;

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
    ) -> (
        ProvingKey<G1Affine>,
        BaseCircuitParams,
        MultiPhaseThreadBreakPoints,
    ) {
        let builder = Self::builder(CircuitBuilderStage::Keygen, k);
        let mut builder = Self::populate(builder, operations, witness);
        builder.calculate_params(Some(20));

        let params = read_params(k as u32);
        let break_points;
        // if pk already exists, read break points from file
        let pk = if Path::new("halo2_final.pk").exists() {
            let file = File::open("halo2_final.json").unwrap();
            break_points = serde_json::from_reader(file).unwrap();
            gen_pk(&params, &builder, Some(Path::new("halo2_final.pk")))
        } else {
            let pk = gen_pk(&params, &builder, Some(Path::new("halo2_final.pk")));
            break_points = builder.break_points();
            pk
        };
        let config_params = builder.config_params.clone();

        let file = File::create("halo2_final.json").unwrap();
        serde_json::to_writer(file, &break_points).unwrap();
        (pk, config_params, break_points)
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

        let params = read_params(k as u32);
        gen_snark_shplonk(&params, pk, builder, None::<&str>)
    }

    pub fn full_prove<
        C: Config<N = Bn254Fr, F = BabyBear, EF = BinomialExtensionField<BabyBear, 4>> + Debug,
    >(
        k: usize,
        operations: TracedVec<DslIr<C>>,
        witness: Witness<C>,
    ) -> Snark {
        let (pk, config_params, break_points) =
            Self::keygen(k, operations.clone(), witness.clone());
        Self::prove(config_params, break_points, &pk, operations, witness)
    }
}

#[cfg(test)]
mod tests {
    use afs_compiler::ir::{Builder, Witness};
    use p3_baby_bear::BabyBear;
    use p3_bn254_fr::Bn254Fr;
    use p3_field::{reduce_32 as reduce_32_gt, split_32 as split_32_gt, AbstractField};

    use crate::{
        config::outer::OuterConfig,
        halo2::Halo2Prover,
        utils::{reduce_32, split_32},
    };

    #[test]
    fn test_num2bits_v() {
        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let mut value_u32 = 1345237507;
        let value = builder.eval(Bn254Fr::from_canonical_u32(value_u32));
        let result = builder.num2bits_v_circuit(value, 32);
        for r in result {
            builder.assert_var_eq(r, Bn254Fr::from_canonical_u32(value_u32 & 1));
            value_u32 >>= 1;
        }

        Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
    }

    #[test]
    fn test_reduce_32() {
        let value_1 = BabyBear::from_canonical_u32(1345237507);
        let value_2 = BabyBear::from_canonical_u32(1000001);
        let gt: Bn254Fr = reduce_32_gt(&[value_1, value_2]);

        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let value_1 = builder.eval(value_1);
        let value_2 = builder.eval(value_2);
        let result = reduce_32(&mut builder, &[value_1, value_2]);
        builder.assert_var_eq(result, gt);

        Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
    }

    #[test]
    fn test_split_32() {
        let value = Bn254Fr::from_canonical_u32(1345237507);
        let gt: Vec<BabyBear> = split_32_gt(value, 3);
        dbg!(&gt);

        let mut builder = Builder::<OuterConfig>::default();
        builder.flags.static_only = true;
        let value = builder.eval(value);
        let result = split_32(&mut builder, value, 3);

        builder.assert_felt_eq(result[0], gt[0]);
        builder.assert_felt_eq(result[1], gt[1]);
        builder.assert_felt_eq(result[2], gt[2]);

        Halo2Prover::mock::<OuterConfig>(10, builder.operations, Witness::default());
    }
}
