use ark_bn254::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::{Field, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use halo2curves_axiom::bn256::{
    Fq as HaloFq, Fq2 as HaloFq2, FROBENIUS_COEFF_FQ6_C1 as HALO_FROBENIUS_COEFF_FQ6_C1,
    FROBENIUS_COEFF_FQ6_C2 as HALO_FROBENIUS_COEFF_FQ6_C2,
    XI_TO_Q_MINUS_1_OVER_2 as HALO_XI_TO_Q_MINUS_1_OVER_2,
};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use openvm_pairing_guest::halo2curves_shims::naf::biguint_to_naf;

use super::{EXP1, EXP2, FQ12_NUM_BYTES, FQ_NUM_BYTES, M_INV, R_INV};
use crate::{
    arkworks::exp_naf,
    bn254::{biguint_to_prime_field, U27_COEFF_0, U27_COEFF_1},
};

lazy_static! {
    static ref EXP1_NAF: Vec<i8> = biguint_to_naf(&EXP1);
    static ref EXP2_NAF: Vec<i8> = biguint_to_naf(&EXP2);
    static ref R_INV_NAF: Vec<i8> = biguint_to_naf(&R_INV);
    static ref M_INV_NAF: Vec<i8> = biguint_to_naf(&M_INV);
    static ref TWENTY_SEVEN_NAF: Vec<i8> = biguint_to_naf(&BigUint::from(27u32));
    static ref UNITY_ROOT_27: Fq12 = {
        let u0 = biguint_to_prime_field::<Fq>(&U27_COEFF_0);
        let u1 = biguint_to_prime_field::<Fq>(&U27_COEFF_1);
        let u_coeffs = Fq2::new(u0, u1);
        Fq12::new(Fq6::new(Fq2::zero(), u_coeffs, Fq2::zero()), Fq6::zero())
    };
    static ref UNITY_ROOT_27_EXP2: Fq12 = exp_naf(&UNITY_ROOT_27, true, &EXP2_NAF);
    pub(crate) static ref FROBENIUS_COEFF_FQ6_C1: [Fq2; 6] = {
        let mut coeffs = [Fq2::zero(); 6];
        for (i, value) in HALO_FROBENIUS_COEFF_FQ6_C1.iter().enumerate() {
            coeffs[i] = halo_fq2_to_ark(value);
        }
        coeffs
    };
    pub(crate) static ref FROBENIUS_COEFF_FQ6_C2: [Fq2; 6] = {
        let mut coeffs = [Fq2::zero(); 6];
        for (i, value) in HALO_FROBENIUS_COEFF_FQ6_C2.iter().enumerate() {
            coeffs[i] = halo_fq2_to_ark(value);
        }
        coeffs
    };
    pub(crate) static ref XI_TO_Q_MINUS_1_OVER_2: Fq2 =
        halo_fq2_to_ark(&HALO_XI_TO_Q_MINUS_1_OVER_2);
}

pub fn parse_g1_points<const N: usize>(raw: Vec<[[u8; N]; 2]>) -> Vec<G1Affine> {
    raw.into_iter()
        .map(|coords| {
            let [x_bytes, y_bytes] = coords;
            let x = Fq::deserialize_uncompressed(&x_bytes[..])
                .expect("Failed to deserialize x coordinate");
            let y = Fq::deserialize_uncompressed(&y_bytes[..])
                .expect("Failed to deserialize y coordinate");
            if x.is_zero() && y.is_zero() {
                G1Affine::identity()
            } else {
                G1Affine::new_unchecked(x, y)
            }
        })
        .collect()
}

pub fn parse_g2_points<const N: usize>(raw: Vec<[[[u8; N]; 2]; 2]>) -> Vec<G2Affine> {
    raw.into_iter()
        .map(|coords| {
            let [[x_c0, x_c1], [y_c0, y_c1]] = coords;
            let x = Fq2::new(
                Fq::deserialize_uncompressed(&x_c0[..]).expect("Failed to deserialize x_c0"),
                Fq::deserialize_uncompressed(&x_c1[..]).expect("Failed to deserialize x_c1"),
            );
            let y = Fq2::new(
                Fq::deserialize_uncompressed(&y_c0[..]).expect("Failed to deserialize y_c0"),
                Fq::deserialize_uncompressed(&y_c1[..]).expect("Failed to deserialize y_c1"),
            );
            if x.is_zero() && y.is_zero() {
                G2Affine::identity()
            } else {
                G2Affine::new_unchecked(x, y)
            }
        })
        .collect()
}

pub fn pairing_hint_bytes(p: &[G1Affine], q: &[G2Affine]) -> Vec<u8> {
    let miller = multi_miller_loop(p, q);
    let (c, u) = final_exp_hint(&miller);
    let mut bytes = Vec::with_capacity(2 * FQ12_NUM_BYTES);
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&c));
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&u));
    bytes
}

fn multi_miller_loop_embedded_exp(p: &[G1Affine], q: &[G2Affine], c: Option<Fq12>) -> Fq12 {
    crate::arkworks::miller_loop_bn254::multi_miller_loop_embedded_exp(p, q, c)
}

fn multi_miller_loop(p: &[G1Affine], q: &[G2Affine]) -> Fq12 {
    multi_miller_loop_embedded_exp(p, q, None)
}

// Compute final exponentiation hint (residue_witness, scaling_factor)
fn final_exp_hint(f: &Fq12) -> (Fq12, Fq12) {
    let unity_root_27 = *UNITY_ROOT_27;
    debug_assert_eq!(exp_naf(&unity_root_27, true, &TWENTY_SEVEN_NAF), Fq12::ONE);

    let (mut residue_hint, cubic_non_residue_power) = if exp_naf(f, true, &EXP1_NAF) == Fq12::ONE {
        (*f, Fq12::ONE)
    } else {
        let f_mul_unity_root_27 = *f * unity_root_27;
        if exp_naf(&f_mul_unity_root_27, true, &EXP1_NAF) == Fq12::ONE {
            (f_mul_unity_root_27, unity_root_27)
        } else {
            (f_mul_unity_root_27 * unity_root_27, unity_root_27.square())
        }
    };

    residue_hint = exp_naf(&residue_hint, true, &R_INV_NAF);
    residue_hint = exp_naf(&residue_hint, true, &M_INV_NAF);
    let mut x = exp_naf(&residue_hint, true, &EXP2_NAF);
    let residue_hint_inv = residue_hint.inverse().unwrap();
    let mut x3 = x.square() * x * residue_hint_inv;
    let mut t = 0;
    let mut tmp = x3.square();

    fn tonelli_shanks_loop(x3: &mut Fq12, tmp: &mut Fq12, t: &mut i32) {
        while *x3 != Fq12::ONE {
            *tmp = (*x3).square();
            *x3 *= *tmp;
            *t += 1;
        }
    }

    tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);
    let unity_root_27_exp2 = *UNITY_ROOT_27_EXP2;
    while t != 0 {
        tmp = unity_root_27_exp2;
        x *= tmp;

        x3 = x.square() * x * residue_hint_inv;
        t = 0;
        tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);
    }

    debug_assert_eq!(residue_hint, x * x * x);
    residue_hint = x;

    (residue_hint, cubic_non_residue_power)
}

/// Converts arkworks Fq12 to bytes in halo2curves layout.
/// This is the format expected by OpenVM guest code.
///
/// halo2curves stores Fq12 as [c0.c0, c1.c0, c0.c1, c1.c1, c0.c2, c1.c2]
/// (interleaving the c0/c1 Fq6 components for each degree of the tower)
fn fq12_to_bytes_halo_layout(value: &Fq12) -> [u8; FQ12_NUM_BYTES] {
    let coeffs = [
        value.c0.c0,
        value.c1.c0,
        value.c0.c1,
        value.c1.c1,
        value.c0.c2,
        value.c1.c2,
    ];

    let mut bytes = [0u8; FQ12_NUM_BYTES];

    coeffs
        .iter()
        .flat_map(|fq2| [fq2.c0, fq2.c1])
        .enumerate()
        .for_each(|(i, limb)| {
            let start = i * FQ_NUM_BYTES;
            let end = start + FQ_NUM_BYTES;
            limb.serialize_uncompressed(&mut bytes[start..end])
                .expect("Failed to serialize Fq");
        });

    bytes
}

fn halo_fq_to_ark(value: &HaloFq) -> Fq {
    let bytes = value.to_bytes();
    Fq::deserialize_uncompressed(&bytes[..]).expect("valid halo fq bytes")
}

fn halo_fq2_to_ark(value: &HaloFq2) -> Fq2 {
    Fq2::new(halo_fq_to_ark(&value.c0), halo_fq_to_ark(&value.c1))
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ec::AffineRepr;
    use ark_ff::UniformRand;
    use halo2curves_axiom::bn256::Fq12 as HaloFq12;
    use openvm_ecc_guest::algebra::field::FieldExtension;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::bn254::halo2curves as bn254_halo;

    fn fq_to_bytes(value: &Fq) -> [u8; FQ_NUM_BYTES] {
        let mut bytes = [0u8; FQ_NUM_BYTES];
        value
            .serialize_uncompressed(&mut bytes[..])
            .expect("Failed to serialize Fq");
        bytes
    }

    fn halo_fq12_to_bytes(value: &HaloFq12) -> Vec<u8> {
        value
            .to_coeffs()
            .into_iter()
            .flat_map(|fq2| fq2.to_coeffs())
            .flat_map(|fq| fq.to_bytes())
            .collect()
    }

    #[test]
    fn test_final_exp_hint_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..3 {
            let ark_g1 = G1Affine::rand(&mut rng);
            let ark_g2 = G2Affine::rand(&mut rng);

            let raw_p = vec![[fq_to_bytes(&ark_g1.x), fq_to_bytes(&ark_g1.y)]];
            let raw_q = vec![[
                [fq_to_bytes(&ark_g2.x.c0), fq_to_bytes(&ark_g2.x.c1)],
                [fq_to_bytes(&ark_g2.y.c0), fq_to_bytes(&ark_g2.y.c1)],
            ]];

            let p_halo = bn254_halo::parse_g1_points(raw_p.clone());
            let q_halo = bn254_halo::parse_g2_points(raw_q.clone());

            let halo_miller = bn254_halo::multi_miller_loop(&p_halo, &q_halo);
            let (c_halo, u_halo) = bn254_halo::final_exp_hint(&halo_miller);

            let ark_miller = multi_miller_loop(&[ark_g1], &[ark_g2]);
            let (c_ark, u_ark) = final_exp_hint(&ark_miller);

            assert_eq!(
                fq12_to_bytes_halo_layout(&c_ark).to_vec(),
                halo_fq12_to_bytes(&c_halo),
                "c hint mismatch with halo2"
            );
            assert_eq!(
                fq12_to_bytes_halo_layout(&u_ark).to_vec(),
                halo_fq12_to_bytes(&u_halo),
                "u hint mismatch with halo2"
            );
        }
    }

    #[test]
    fn test_pairing_hint_bytes_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..3 {
            let ark_g1 = G1Affine::rand(&mut rng);
            let ark_g2 = G2Affine::rand(&mut rng);

            let raw_p = vec![[fq_to_bytes(&ark_g1.x), fq_to_bytes(&ark_g1.y)]];
            let raw_q = vec![[
                [fq_to_bytes(&ark_g2.x.c0), fq_to_bytes(&ark_g2.x.c1)],
                [fq_to_bytes(&ark_g2.y.c0), fq_to_bytes(&ark_g2.y.c1)],
            ]];

            let p_ark = parse_g1_points(raw_p.clone());
            let q_ark = parse_g2_points(raw_q.clone());
            let bytes_ark = pairing_hint_bytes(&p_ark, &q_ark);

            let p_halo = bn254_halo::parse_g1_points(raw_p);
            let q_halo = bn254_halo::parse_g2_points(raw_q);
            let bytes_halo = bn254_halo::pairing_hint_bytes(&p_halo, &q_halo);

            assert_eq!(
                bytes_ark, bytes_halo,
                "Pairing hint bytes mismatch with halo2"
            );
        }
    }

    #[test]
    fn test_pairing_check() {
        let a = [Fr::from(1), Fr::from(2)];
        let b = [Fr::from(2), Fr::from(1)];

        let mut p_vec = Vec::new();
        let mut q_vec = Vec::new();
        for i in 0..2 {
            let p = ark_ec::short_weierstrass::Affine::generator() * a[i];
            let mut p = G1Affine::from(p);
            if i % 2 == 1 {
                p.y = -p.y;
            }
            let q = ark_ec::short_weierstrass::Affine::generator() * b[i];
            let q = G2Affine::from(q);
            p_vec.push(p);
            q_vec.push(q);
        }

        let miller = multi_miller_loop(&p_vec, &q_vec);
        let (c, u) = final_exp_hint(&miller);

        // We follow Theorem 3 of https://eprint.iacr.org/2024/640.pdf to check that the pairing equals 1
        // By the theorem, it suffices to provide c and u such that f * u == c^λ.
        // Since λ = 6x + 2 + q^3 - q^2 + q, we will check the equivalent condition:
        // f * c^-{6x + 2} * u * c^-{q^3 - q^2 + q} == 1

        let c_inv = c.inverse().unwrap();

        // c_mul = c^-{q^3 - q^2 + q}
        let mut c_q3_inv = c_inv;
        c_q3_inv.frobenius_map_in_place(3);
        let mut c_q2 = c;
        c_q2.frobenius_map_in_place(2);
        let mut c_q_inv = c_inv;
        c_q_inv.frobenius_map_in_place(1);
        let c_mul = c_q3_inv * c_q2 * c_q_inv;

        // Pass c inverse into the miller loop so that we compute fc == f * c^-{6x + 2}
        let fc = multi_miller_loop_embedded_exp(&p_vec, &q_vec, Some(c_inv));

        assert_eq!(fc * c_mul * u, Fq12::ONE, "Pairing check failed");
    }
}
