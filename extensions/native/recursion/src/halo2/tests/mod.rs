use std::io::Write;

use openvm_native_compiler::{
    constraints::halo2::compiler::convert_fr,
    ir::{Builder, Witness},
};
use openvm_stark_backend::p3_field::{
    reduce_32 as reduce_32_gt, split_32 as split_32_gt, AbstractField,
};
use openvm_stark_sdk::{p3_baby_bear::BabyBear, p3_bn254_fr::Bn254Fr};
use snark_verifier_sdk::{
    halo2::{gen_dummy_snark_from_vk, gen_snark_shplonk},
    snark_verifier::{
        halo2_base::{
            gates::circuit::{builder::BaseCircuitBuilder, CircuitBuilderStage::Keygen},
            halo2_proofs::{halo2curves::bn256::Fr, plonk::keygen_pk2},
        },
        util::arithmetic::Field,
    },
    Snark, SHPLONK,
};

use crate::{
    config::outer::OuterConfig,
    halo2::{
        utils::gen_kzg_params, wrapper::Halo2WrapperProvingKey, CircuitBuilderStage::Prover,
        DslOperations, Halo2Prover, Halo2ProvingMetadata, Halo2ProvingPinning,
    },
    utils::{reduce_32, split_32},
};

mod multi_field32;
mod outer_poseidon2;
mod stark;

const DUMMY_K: usize = 10;
const DUMMY_N: usize = 2 * (1 << DUMMY_K);
fn build_dummy_circuit(builder: &mut BaseCircuitBuilder<Fr>, n: usize) {
    let ctx = builder.main(0);
    let zero = ctx.load_constant(Field::ZERO);
    for _ in 0..n {
        ctx.load_witness(Field::ZERO);
    }
    builder.assigned_instances = vec![vec![zero]];
}
/// Return (dummy snark, real snark, pinning)
fn snarks_dummy_circuit() -> (Snark, Snark, Halo2ProvingPinning) {
    let k = DUMMY_K;
    let n = DUMMY_N;
    let params = gen_kzg_params(k as u32);
    let mut builder = BaseCircuitBuilder::from_stage(Keygen)
        .use_k(k)
        .use_instance_columns(1);
    build_dummy_circuit(&mut builder, n);
    builder.calculate_params(Some(20));
    let config_params = builder.config_params.clone();
    let pk = keygen_pk2(&params, &builder, false).unwrap();
    let break_points = builder.break_points();
    let dummy_snark = gen_dummy_snark_from_vk::<SHPLONK>(&params, pk.get_vk(), vec![1], None);
    let mut builder = BaseCircuitBuilder::from_stage(Prover)
        .use_params(config_params.clone())
        .use_break_points(break_points.clone());
    build_dummy_circuit(&mut builder, n);
    let snark = gen_snark_shplonk(&params, &pk, builder, None::<&str>);
    let pinning = Halo2ProvingPinning {
        pk,
        metadata: Halo2ProvingMetadata {
            config_params,
            break_points,
            num_pvs: vec![1],
        },
    };
    (dummy_snark, snark, pinning)
}

#[test]
fn test_publish() {
    let mut builder = Builder::<OuterConfig>::default();
    builder.flags.static_only = true;
    let value_u32 = 1345237507;
    let value_fr = Bn254Fr::from_canonical_u32(value_u32);
    let value = builder.eval(value_fr);
    builder.static_commit_public_value(0, value);

    let pis = Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 1,
        },
        Witness::default(),
    );
    assert_eq!(pis, vec![vec![convert_fr(&value_fr)]]);
}

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

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
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

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
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

    Halo2Prover::mock::<OuterConfig>(
        10,
        DslOperations {
            operations: builder.operations,
            num_public_values: 0,
        },
        Witness::default(),
    );
}

#[test]
fn test_wrapper_select_k() {
    let (dummy_snark, _, _) = snarks_dummy_circuit();
    let wrapper_k = Halo2WrapperProvingKey::select_k(dummy_snark);
    assert_eq!(wrapper_k, 22);
}

#[test]
fn test_pinning_serde() {
    let (_, _, pinning) = snarks_dummy_circuit();
    // Something went wrong when Halo2ProvingPinning is a field. So we explicitly test with a struct
    // contains Haplo2ProvingPinning.
    let wrapper = Halo2WrapperProvingKey {
        pinning: pinning.clone(),
    };

    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(&bitcode::serialize(&wrapper).unwrap()).unwrap();
    let new_wrapper: Halo2WrapperProvingKey =
        bitcode::deserialize(&std::fs::read(f.path()).unwrap()).unwrap();
    let new_pinning = new_wrapper.pinning;
    let params = gen_kzg_params(DUMMY_K as u32);
    let mut builder = BaseCircuitBuilder::from_stage(Prover)
        .use_params(new_pinning.metadata.config_params.clone())
        .use_break_points(new_pinning.metadata.break_points.clone());
    build_dummy_circuit(&mut builder, DUMMY_N);
    gen_snark_shplonk(&params, &pinning.pk, builder, None::<&str>);
}
