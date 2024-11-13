#![allow(non_snake_case)]

use ax_ecc_execution::{
    axvm_ecc::{
        curve::bls12381::{G1Affine, G2Affine},
        field::FieldExtension,
        pairing::{FinalExp, MultiMillerLoop},
        point::AffinePoint,
    },
    curves::bls12_381::Bls12_381,
};
use ax_stark_sdk::ax_stark_backend::p3_field::AbstractField;
use axvm_circuit::arch::{VmConfig, VmExecutor};
use eyre::Result;
use p3_baby_bear::BabyBear;

use crate::utils::build_example_program;

type F = BabyBear;

#[test]
fn test_bls12_381_final_exp_hint() -> Result<()> {
    let elf = build_example_program("final_exp_hint")?;
    let executor = VmExecutor::<F>::new(VmConfig::rv32im());

    let bls12_381 = Bls12_381;
    let P = G1Affine::generator();
    let Q = G2Affine::generator();
    let ps = vec![AffinePoint::new(P.x, P.y), AffinePoint::new(P.x, -P.y)];
    let qs = vec![AffinePoint::new(Q.x, Q.y), AffinePoint::new(Q.x, Q.y)];
    let f = bls12_381.multi_miller_loop(&ps, &qs);
    let (c, s) = bls12_381.final_exp_hint(f);
    let io = [f, c, s]
        .into_iter()
        .flat_map(|fp12| fp12.to_coeffs())
        .flat_map(|fp2| fp2.to_coeffs())
        .flat_map(|fp| fp.to_bytes())
        .map(AbstractField::from_canonical_u8)
        .collect::<Vec<_>>();
    executor.execute(elf, vec![io])?;
    Ok(())
}
