mod multi_field32;
mod outer_poseidon2;
mod stark;

use axvm_native_compiler::{
    constraints::halo2::compiler::convert_fr,
    ir::{Builder, Witness},
};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::{reduce_32 as reduce_32_gt, split_32 as split_32_gt, AbstractField};

use crate::{
    config::outer::OuterConfig,
    halo2::{DslOperations, Halo2Prover},
    utils::{reduce_32, split_32},
};

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
