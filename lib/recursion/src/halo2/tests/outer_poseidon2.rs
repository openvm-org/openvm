use ax_stark_sdk::config::baby_bear_poseidon2_outer::outer_perm;
use axvm_native_compiler::ir::{Builder, Felt, Var, Witness};
use p3_baby_bear::BabyBear;
use p3_bn254_fr::Bn254Fr;
use p3_field::AbstractField;
use p3_symmetric::{CryptographicHasher, Permutation, PseudoCompressionFunction};

use crate::{
    config::outer::{OuterCompress, OuterConfig, OuterHash},
    halo2::Halo2Prover,
    outer_poseidon2::Poseidon2CircuitBuilder,
    vars::OuterDigestVariable,
};

#[test]
fn test_p2_permute_mut() {
    let poseidon2 = outer_perm();
    let input: [Bn254Fr; 3] = [
        Bn254Fr::from_canonical_u32(0),
        Bn254Fr::from_canonical_u32(1),
        Bn254Fr::from_canonical_u32(2),
    ];
    let mut output = input;
    poseidon2.permute_mut(&mut output);

    let mut builder = Builder::<OuterConfig>::default();
    let a: Var<_> = builder.eval(input[0]);
    let b: Var<_> = builder.eval(input[1]);
    let c: Var<_> = builder.eval(input[2]);
    builder.p2_permute_mut([a, b, c]);

    builder.assert_var_eq(a, output[0]);
    builder.assert_var_eq(b, output[1]);
    builder.assert_var_eq(c, output[2]);

    Halo2Prover::mock(10, builder.operations, Witness::default());
}

#[test]
fn test_p2_hash() {
    let perm = outer_perm();
    let hasher = OuterHash::new(perm.clone()).unwrap();

    let input: [BabyBear; 7] = [
        BabyBear::from_canonical_u32(0),
        BabyBear::from_canonical_u32(1),
        BabyBear::from_canonical_u32(2),
        BabyBear::from_canonical_u32(2),
        BabyBear::from_canonical_u32(2),
        BabyBear::from_canonical_u32(2),
        BabyBear::from_canonical_u32(2),
    ];
    let output = hasher.hash_iter(input);

    let mut builder = Builder::<OuterConfig>::default();
    let a: Felt<_> = builder.eval(input[0]);
    let b: Felt<_> = builder.eval(input[1]);
    let c: Felt<_> = builder.eval(input[2]);
    let d: Felt<_> = builder.eval(input[3]);
    let e: Felt<_> = builder.eval(input[4]);
    let f: Felt<_> = builder.eval(input[5]);
    let g: Felt<_> = builder.eval(input[6]);
    let result = builder.p2_hash(&[a, b, c, d, e, f, g]);
    builder.assert_var_eq(result[0], output[0]);

    Halo2Prover::mock(10, builder.operations, Witness::default());
}

#[test]
fn test_p2_compress() {
    let perm = outer_perm();
    let compressor = OuterCompress::new(perm.clone());

    let a: [Bn254Fr; 1] = [Bn254Fr::two()];
    let b: [Bn254Fr; 1] = [Bn254Fr::two()];
    let gt = compressor.compress([a, b]);

    let mut builder = Builder::<OuterConfig>::default();
    let a: OuterDigestVariable<OuterConfig> = [builder.eval(a[0])];
    let b: OuterDigestVariable<OuterConfig> = [builder.eval(b[0])];
    let result = builder.p2_compress([a, b]);
    builder.assert_var_eq(result[0], gt[0]);

    Halo2Prover::mock(10, builder.operations, Witness::default());
}
