use halo2curves_axiom::bn256::{Fq, Fq12, Fq2};
use halo2curves_axiom::ff::PrimeField;
use openvm_ecc_guest::{algebra::field::FieldExtension, AffinePoint};
use openvm_pairing_guest::{
    halo2curves_shims::bn254::Bn254,
    pairing::{FinalExp, MultiMillerLoop},
};

use super::{FQ12_NUM_BYTES, FQ_NUM_BYTES};

pub type G1Affine = AffinePoint<Fq>;
pub type G2Affine = AffinePoint<Fq2>;

pub fn parse_g1_points<const N: usize>(raw: Vec<[[u8; N]; 2]>) -> Vec<G1Affine>
where
    <Fq as PrimeField>::Repr: From<[u8; N]>,
{
    raw.into_iter()
        .map(|coords| {
            let [x_bytes, y_bytes] = coords;
            let x = bytes_to_field::<Fq, N>(x_bytes);
            let y = bytes_to_field::<Fq, N>(y_bytes);
            AffinePoint::new(x, y)
        })
        .collect()
}

pub fn parse_g2_points<const N: usize>(raw: Vec<[[[u8; N]; 2]; 2]>) -> Vec<G2Affine>
where
    <Fq as PrimeField>::Repr: From<[u8; N]>,
{
    raw.into_iter()
        .map(|coords| {
            let [[x_c0, x_c1], [y_c0, y_c1]] = coords;
            let x = Fq2 {
                c0: bytes_to_field::<Fq, N>(x_c0),
                c1: bytes_to_field::<Fq, N>(x_c1),
            };
            let y = Fq2 {
                c0: bytes_to_field::<Fq, N>(y_c0),
                c1: bytes_to_field::<Fq, N>(y_c1),
            };
            AffinePoint::new(x, y)
        })
        .collect()
}

pub fn multi_miller_loop(p: &[G1Affine], q: &[G2Affine]) -> Fq12 {
    Bn254::multi_miller_loop(p, q)
}

pub fn final_exp_hint(f: &Fq12) -> (Fq12, Fq12) {
    Bn254::final_exp_hint(f)
}

pub fn pairing_hint_bytes(p: &[G1Affine], q: &[G2Affine]) -> Vec<u8> {
    let miller = multi_miller_loop(p, q);
    let (c, u) = final_exp_hint(&miller);
    let mut bytes = Vec::with_capacity(2 * FQ12_NUM_BYTES);
    bytes.extend_from_slice(&fq12_to_bytes(&c));
    bytes.extend_from_slice(&fq12_to_bytes(&u));
    bytes
}

pub fn fq12_to_bytes(value: &Fq12) -> [u8; FQ12_NUM_BYTES] {
    let mut bytes = [0u8; FQ12_NUM_BYTES];
    let mut offset = 0;
    for fp2 in value.to_coeffs() {
        for fp in fp2.to_coeffs() {
            let limb_bytes = fp.to_bytes();
            bytes[offset..offset + FQ_NUM_BYTES].copy_from_slice(&limb_bytes);
            offset += FQ_NUM_BYTES;
        }
    }
    bytes
}

fn bytes_to_field<F: PrimeField, const N: usize>(bytes: [u8; N]) -> F
where
    <F as PrimeField>::Repr: From<[u8; N]>,
{
    let repr = <F as PrimeField>::Repr::from(bytes);
    F::from_repr(repr).expect("invalid field element bytes")
}
