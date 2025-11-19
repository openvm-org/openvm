use super::{
    biguint_to_prime_field, EXP1_LIMBS, EXP2_LIMBS, FQ12_NUM_BYTES, FQ2_NUM_BYTES, FQ6_NUM_BYTES,
    FQ_NUM_BYTES, M_INV_LIMBS, R_INV_LIMBS, U27_COEFF_0, U27_COEFF_1,
};

use ark_bn254::{Bn254 as ArkBn254, Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ec::pairing::{prepare_g1, prepare_g2, Pairing};
use ark_ff::{Field, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use lazy_static::lazy_static;

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
    let (c, u) = final_exp_witness(&miller);
    let mut bytes = Vec::with_capacity(2 * FQ12_NUM_BYTES);
    bytes.extend_from_slice(&fq12_to_bytes(&c));
    bytes.extend_from_slice(&fq12_to_bytes(&u));
    bytes
}

pub fn multi_miller_loop(p: &[G1Affine], q: &[G2Affine]) -> Fq12 {
    let g1_it = p.iter().map(|point| prepare_g1::<ArkBn254>(*point));
    let g2_it = q.iter().map(|point| prepare_g2::<ArkBn254>(*point));
    ArkBn254::multi_miller_loop(g1_it, g2_it).0
}

lazy_static! {
    static ref UNITY_ROOT_27: Fq12 = {
        let u0 = biguint_to_prime_field::<Fq>(&U27_COEFF_0);
        let u1 = biguint_to_prime_field::<Fq>(&U27_COEFF_1);
        let u_coeffs = Fq2::new(u0, u1);
        Fq12::new(Fq6::new(Fq2::zero(), Fq2::zero(), u_coeffs), Fq6::zero())
    };
    static ref UNITY_ROOT_27_EXP2: Fq12 = UNITY_ROOT_27.pow(EXP2_LIMBS.as_slice());
}

pub fn final_exp_witness(f: &Fq12) -> (Fq12, Fq12) {
    let unity_root_27 = *UNITY_ROOT_27;
    debug_assert_eq!(unity_root_27.pow([27u64]), Fq12::ONE);

    let (mut residue_witness, cubic_non_residue_power) =
        if f.pow(EXP1_LIMBS.as_slice()) == Fq12::ONE {
            (*f, Fq12::ONE)
        } else {
            let f_mul_unity_root_27 = *f * unity_root_27;
            if f_mul_unity_root_27.pow(EXP1_LIMBS.as_slice()) == Fq12::ONE {
                (f_mul_unity_root_27, unity_root_27)
            } else {
                (f_mul_unity_root_27 * unity_root_27, unity_root_27.square())
            }
        };

    residue_witness = residue_witness.pow(R_INV_LIMBS.as_slice());
    residue_witness = residue_witness.pow(M_INV_LIMBS.as_slice());
    let mut x = residue_witness.pow(EXP2_LIMBS.as_slice());
    let residue_witness_inv = residue_witness.inverse().unwrap();
    let mut x3 = x.square() * x * residue_witness_inv;
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

        x3 = x.square() * x * residue_witness_inv;
        t = 0;
        tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);
    }

    debug_assert_eq!(residue_witness, x * x * x);
    residue_witness = x;

    (residue_witness, cubic_non_residue_power)
}

fn fq12_to_bytes(value: &Fq12) -> [u8; FQ12_NUM_BYTES] {
    let mut bytes = [0u8; FQ12_NUM_BYTES];
    value
        .serialize_uncompressed(&mut bytes[..])
        .expect("Failed to serialize Fq12");
    bytes
}

fn fq_to_bytes(value: &Fq) -> [u8; FQ_NUM_BYTES] {
    let mut bytes_le = [0u8; FQ_NUM_BYTES];
    value
        .serialize_uncompressed(&mut bytes_le[..])
        .expect("Failed to serialize Fq");
    bytes_le
}

fn fq2_to_bytes(value: &Fq2) -> [u8; FQ2_NUM_BYTES] {
    let mut bytes = [0u8; FQ2_NUM_BYTES];
    value
        .serialize_uncompressed(&mut bytes[..])
        .expect("Failed to serialize Fq2");
    bytes
}

fn fq6_to_bytes(value: &Fq6) -> [u8; FQ6_NUM_BYTES] {
    let mut bytes = [0u8; FQ6_NUM_BYTES];
    value
        .serialize_uncompressed(&mut bytes[..])
        .expect("Failed to serialize Fq6");
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::UniformRand;
    use halo2curves_axiom::bn256::{
        Fq as HaloFq, Fq12 as HaloFq12, Fq2 as HaloFq2, Fq6 as HaloFq6, G1Affine as HaloG1,
        G2Affine as HaloG2,
    };
    use openvm_ecc_guest::algebra::field::FieldExtension;
    use openvm_ecc_guest::AffinePoint;
    use openvm_pairing_guest::{
        halo2curves_shims::bn254::Bn254 as GuestBn254,
        pairing::{FinalExp, MultiMillerLoop},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::convert::TryInto;

    fn ark_fq_to_halo(value: &Fq) -> HaloFq {
        HaloFq::from_bytes(&fq_to_bytes(value)).unwrap()
    }

    fn ark_fp2_to_halo(value: &Fq2) -> HaloFq2 {
        HaloFq2 {
            c0: ark_fq_to_halo(&value.c0),
            c1: ark_fq_to_halo(&value.c1),
        }
    }

    fn halo_fq_to_ark(value: &HaloFq) -> Fq {
        let bytes = value.to_bytes();
        Fq::deserialize_uncompressed(&bytes[..]).unwrap()
    }

    fn halo_fp2_to_ark(value: &HaloFq2) -> Fq2 {
        Fq2::new(halo_fq_to_ark(&value.c0), halo_fq_to_ark(&value.c1))
    }

    fn ark_fq12_to_halo(value: &Fq12) -> HaloFq12 {
        HaloFq12::from_coeffs([
            ark_fp2_to_halo(&value.c0.c0),
            ark_fp2_to_halo(&value.c0.c1),
            ark_fp2_to_halo(&value.c0.c2),
            ark_fp2_to_halo(&value.c1.c0),
            ark_fp2_to_halo(&value.c1.c1),
            ark_fp2_to_halo(&value.c1.c2),
        ])
    }

    fn halo_g1_to_ark(g: &HaloG1) -> G1Affine {
        G1Affine::new_unchecked(halo_fq_to_ark(&g.x), halo_fq_to_ark(&g.y))
    }

    fn halo_g2_to_ark(g: &HaloG2) -> G2Affine {
        G2Affine::new_unchecked(halo_fp2_to_ark(&g.x), halo_fp2_to_ark(&g.y))
    }

    fn halo_g1_to_affine_point(g: &HaloG1) -> AffinePoint<HaloFq> {
        AffinePoint::new(g.x.clone(), g.y.clone())
    }

    fn halo_g2_to_affine_point(g: &HaloG2) -> AffinePoint<HaloFq2> {
        AffinePoint::new(g.x.clone(), g.y.clone())
    }

    fn halo_fq12_to_bytes(value: &HaloFq12) -> Vec<u8> {
        value
            .to_coeffs()
            .into_iter()
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect()
    }

    fn ark_fq12_from_bytes(bytes: &[u8]) -> Fq12 {
        assert_eq!(bytes.len(), FQ12_NUM_BYTES);
        let mut fp2_coeffs = Vec::with_capacity(6);
        let mut offset = 0;
        for _ in 0..6 {
            let c0 = Fq::deserialize_uncompressed(&bytes[offset..offset + FQ_NUM_BYTES]).unwrap();
            offset += FQ_NUM_BYTES;
            let c1 = Fq::deserialize_uncompressed(&bytes[offset..offset + FQ_NUM_BYTES]).unwrap();
            offset += FQ_NUM_BYTES;
            fp2_coeffs.push(Fq2::new(c0, c1));
        }
        Fq12::new(
            Fq6::new(fp2_coeffs[0], fp2_coeffs[1], fp2_coeffs[2]),
            Fq6::new(fp2_coeffs[3], fp2_coeffs[4], fp2_coeffs[5]),
        )
    }

    fn halo_fq12_from_bytes(bytes: &[u8]) -> HaloFq12 {
        assert_eq!(bytes.len(), FQ12_NUM_BYTES);
        let mut fp2_coeffs = Vec::with_capacity(6);
        let mut offset = 0;
        for _ in 0..6 {
            let c0_bytes: [u8; FQ_NUM_BYTES] =
                bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
            offset += FQ_NUM_BYTES;
            let c1_bytes: [u8; FQ_NUM_BYTES] =
                bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
            offset += FQ_NUM_BYTES;
            let c0 = HaloFq::from_bytes(&c0_bytes).unwrap();
            let c1 = HaloFq::from_bytes(&c1_bytes).unwrap();
            fp2_coeffs.push(HaloFq2::new(c0, c1));
        }
        HaloFq12::from_coeffs([
            fp2_coeffs[0],
            fp2_coeffs[1],
            fp2_coeffs[2],
            fp2_coeffs[3],
            fp2_coeffs[4],
            fp2_coeffs[5],
        ])
    }

    fn ark_fq_from_bytes(bytes: &[u8]) -> Fq {
        assert_eq!(bytes.len(), FQ_NUM_BYTES);
        Fq::deserialize_uncompressed(bytes).unwrap()
    }

    fn halo_fq_from_bytes(bytes: &[u8]) -> HaloFq {
        assert_eq!(bytes.len(), FQ_NUM_BYTES);
        let bytes_array: [u8; FQ_NUM_BYTES] = bytes.try_into().unwrap();
        HaloFq::from_bytes(&bytes_array).unwrap()
    }

    fn halo_fq_to_bytes(value: &HaloFq) -> Vec<u8> {
        value.to_bytes().to_vec()
    }

    fn ark_fq2_from_bytes(bytes: &[u8]) -> Fq2 {
        assert_eq!(bytes.len(), FQ2_NUM_BYTES);
        Fq2::deserialize_uncompressed(bytes).unwrap()
    }

    fn halo_fq2_from_bytes(bytes: &[u8]) -> HaloFq2 {
        assert_eq!(bytes.len(), FQ2_NUM_BYTES);
        let mut offset = 0;
        let c0_bytes: [u8; FQ_NUM_BYTES] = bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
        offset += FQ_NUM_BYTES;
        let c1_bytes: [u8; FQ_NUM_BYTES] = bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
        let c0 = HaloFq::from_bytes(&c0_bytes).unwrap();
        let c1 = HaloFq::from_bytes(&c1_bytes).unwrap();
        HaloFq2::new(c0, c1)
    }

    fn halo_fq2_to_bytes(value: &HaloFq2) -> Vec<u8> {
        value
            .to_coeffs()
            .into_iter()
            .flat_map(|fp| fp.to_bytes())
            .collect()
    }

    fn ark_fq6_from_bytes(bytes: &[u8]) -> Fq6 {
        assert_eq!(bytes.len(), FQ6_NUM_BYTES);
        Fq6::deserialize_uncompressed(bytes).unwrap()
    }

    fn halo_fq6_from_bytes(bytes: &[u8]) -> HaloFq6 {
        assert_eq!(bytes.len(), FQ6_NUM_BYTES);
        let mut fp2_coeffs = Vec::with_capacity(3);
        let mut offset = 0;
        for _ in 0..3 {
            let c0_bytes: [u8; FQ_NUM_BYTES] =
                bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
            offset += FQ_NUM_BYTES;
            let c1_bytes: [u8; FQ_NUM_BYTES] =
                bytes[offset..offset + FQ_NUM_BYTES].try_into().unwrap();
            offset += FQ_NUM_BYTES;
            let c0 = HaloFq::from_bytes(&c0_bytes).unwrap();
            let c1 = HaloFq::from_bytes(&c1_bytes).unwrap();
            fp2_coeffs.push(HaloFq2::new(c0, c1));
        }
        HaloFq6 {
            c0: fp2_coeffs[0],
            c1: fp2_coeffs[1],
            c2: fp2_coeffs[2],
        }
    }

    fn halo_fq6_to_bytes(value: &HaloFq6) -> Vec<u8> {
        [value.c0, value.c1, value.c2]
            .into_iter()
            .flat_map(|fp2| fp2.to_coeffs())
            .flat_map(|fp| fp.to_bytes())
            .collect()
    }

    #[test]
    fn test_byte_array_fq_roundtrip() {
        let mut rng = StdRng::seed_from_u64(4242);
        for _ in 0..5 {
            let lhs = Fq::rand(&mut rng);
            let rhs = Fq::rand(&mut rng);

            let lhs_bytes = fq_to_bytes(&lhs);
            let rhs_bytes = fq_to_bytes(&rhs);

            let ark_lhs = ark_fq_from_bytes(&lhs_bytes);
            let ark_rhs = ark_fq_from_bytes(&rhs_bytes);

            assert_eq!(ark_lhs, lhs, "ark lhs mismatch");
            assert_eq!(ark_rhs, rhs, "ark rhs mismatch");

            let halo_lhs = halo_fq_from_bytes(&lhs_bytes);
            let halo_rhs = halo_fq_from_bytes(&rhs_bytes);

            assert_eq!(
                fq_to_bytes(&ark_lhs).to_vec(),
                halo_fq_to_bytes(&halo_lhs),
                "lhs mismatch"
            );
            assert_eq!(
                fq_to_bytes(&ark_rhs).to_vec(),
                halo_fq_to_bytes(&halo_rhs),
                "rhs mismatch"
            );

            let ark_sum = ark_lhs + ark_rhs;
            let halo_sum = halo_lhs + halo_rhs;

            assert_eq!(
                fq_to_bytes(&ark_sum).to_vec(),
                halo_fq_to_bytes(&halo_sum),
                "sum mismatch"
            );

            let ark_double = ark_sum.double();
            let halo_double = halo_sum.double();

            assert_eq!(
                fq_to_bytes(&ark_double).to_vec(),
                halo_fq_to_bytes(&halo_double),
                "double mismatch"
            );

            let ark_square = ark_sum.square();
            let halo_square = halo_sum.square();

            assert_eq!(
                fq_to_bytes(&ark_square).to_vec(),
                halo_fq_to_bytes(&halo_square),
                "square mismatch"
            );

            let ark_mul = ark_lhs * ark_rhs;
            let halo_mul = halo_lhs * halo_rhs;

            assert_eq!(
                fq_to_bytes(&ark_mul).to_vec(),
                halo_fq_to_bytes(&halo_mul),
                "mul mismatch"
            );
        }
    }

    #[test]
    fn test_byte_array_fq2_roundtrip() {
        let mut rng = StdRng::seed_from_u64(4242);
        for _ in 0..5 {
            let lhs = Fq2::rand(&mut rng);
            let rhs = Fq2::rand(&mut rng);

            let lhs_bytes = fq2_to_bytes(&lhs);
            let rhs_bytes = fq2_to_bytes(&rhs);

            let ark_lhs = ark_fq2_from_bytes(&lhs_bytes);
            let ark_rhs = ark_fq2_from_bytes(&rhs_bytes);

            assert_eq!(ark_lhs, lhs, "ark lhs mismatch");
            assert_eq!(ark_rhs, rhs, "ark rhs mismatch");

            let halo_lhs = halo_fq2_from_bytes(&lhs_bytes);
            let halo_rhs = halo_fq2_from_bytes(&rhs_bytes);

            assert_eq!(
                fq2_to_bytes(&ark_lhs).to_vec(),
                halo_fq2_to_bytes(&halo_lhs),
                "lhs mismatch"
            );
            assert_eq!(
                fq2_to_bytes(&ark_rhs).to_vec(),
                halo_fq2_to_bytes(&halo_rhs),
                "rhs mismatch"
            );

            let ark_sum = ark_lhs + ark_rhs;
            let halo_sum = halo_lhs + halo_rhs;

            assert_eq!(
                fq2_to_bytes(&ark_sum).to_vec(),
                halo_fq2_to_bytes(&halo_sum),
                "sum mismatch"
            );

            let ark_double = ark_sum.double();
            let halo_double = halo_sum.double();

            assert_eq!(
                fq2_to_bytes(&ark_double).to_vec(),
                halo_fq2_to_bytes(&halo_double),
                "double mismatch"
            );

            let ark_square = ark_sum.square();
            let halo_square = halo_sum.square();

            assert_eq!(
                fq2_to_bytes(&ark_square).to_vec(),
                halo_fq2_to_bytes(&halo_square),
                "square mismatch"
            );

            let ark_mul = ark_lhs * ark_rhs;
            let halo_mul = halo_lhs * halo_rhs;

            assert_eq!(
                fq2_to_bytes(&ark_mul).to_vec(),
                halo_fq2_to_bytes(&halo_mul),
                "mul mismatch"
            );
        }
    }

    #[test]
    fn test_byte_array_fq6_roundtrip() {
        let mut rng = StdRng::seed_from_u64(4242);
        for _ in 0..5 {
            let lhs = Fq6::rand(&mut rng);
            let rhs = Fq6::rand(&mut rng);

            let lhs_bytes = fq6_to_bytes(&lhs);
            let rhs_bytes = fq6_to_bytes(&rhs);

            let ark_lhs = ark_fq6_from_bytes(&lhs_bytes);
            let ark_rhs = ark_fq6_from_bytes(&rhs_bytes);

            assert_eq!(ark_lhs, lhs, "ark lhs mismatch");
            assert_eq!(ark_rhs, rhs, "ark rhs mismatch");

            let halo_lhs = halo_fq6_from_bytes(&lhs_bytes);
            let halo_rhs = halo_fq6_from_bytes(&rhs_bytes);

            assert_eq!(
                fq6_to_bytes(&ark_lhs).to_vec(),
                halo_fq6_to_bytes(&halo_lhs),
                "lhs mismatch"
            );
            assert_eq!(
                fq6_to_bytes(&ark_rhs).to_vec(),
                halo_fq6_to_bytes(&halo_rhs),
                "rhs mismatch"
            );

            let ark_sum = ark_lhs + ark_rhs;
            let halo_sum = halo_lhs + halo_rhs;

            assert_eq!(
                fq6_to_bytes(&ark_sum).to_vec(),
                halo_fq6_to_bytes(&halo_sum),
                "sum mismatch"
            );

            let ark_double = ark_sum.double();
            let halo_double = halo_sum.double();

            assert_eq!(
                fq6_to_bytes(&ark_double).to_vec(),
                halo_fq6_to_bytes(&halo_double),
                "double mismatch"
            );

            let ark_square = ark_sum.square();
            let halo_square = halo_sum.square();

            assert_eq!(
                fq6_to_bytes(&ark_square).to_vec(),
                halo_fq6_to_bytes(&halo_square),
                "square mismatch"
            );

            let ark_mul = ark_lhs * ark_rhs;
            let halo_mul = halo_lhs * halo_rhs;

            assert_eq!(
                fq6_to_bytes(&ark_mul).to_vec(),
                halo_fq6_to_bytes(&halo_mul),
                "mul mismatch"
            );
        }
    }

    #[test]
    fn test_byte_array_fq12_roundtrip() {
        let mut rng = StdRng::seed_from_u64(4242);
        for _ in 0..5 {
            let lhs = Fq12::rand(&mut rng);
            let rhs = Fq12::rand(&mut rng);

            let lhs_bytes = fq12_to_bytes(&lhs);
            let rhs_bytes = fq12_to_bytes(&rhs);

            let ark_lhs = ark_fq12_from_bytes(&lhs_bytes);
            let ark_rhs = ark_fq12_from_bytes(&rhs_bytes);

            assert_eq!(ark_lhs, lhs, "ark lhs mismatch");
            assert_eq!(ark_rhs, rhs, "ark rhs mismatch");

            let halo_lhs = halo_fq12_from_bytes(&lhs_bytes);
            let halo_rhs = halo_fq12_from_bytes(&rhs_bytes);

            assert_eq!(
                fq12_to_bytes(&ark_lhs).to_vec(),
                halo_fq12_to_bytes(&halo_lhs),
                "lhs mismatch"
            );
            assert_eq!(
                fq12_to_bytes(&ark_rhs).to_vec(),
                halo_fq12_to_bytes(&halo_rhs),
                "rhs mismatch"
            );

            let ark_sum = ark_lhs + ark_rhs;
            let halo_sum = halo_lhs + halo_rhs;

            assert_eq!(
                fq12_to_bytes(&ark_sum).to_vec(),
                halo_fq12_to_bytes(&halo_sum),
                "sum mismatch"
            );

            let ark_double = ark_sum.double();
            let halo_double = halo_sum.double();

            assert_eq!(
                fq12_to_bytes(&ark_double).to_vec(),
                halo_fq12_to_bytes(&halo_double),
                "double mismatch"
            );

            let ark_square = ark_sum.square();
            let halo_square = halo_sum.square();

            assert_eq!(
                fq12_to_bytes(&ark_square).to_vec(),
                halo_fq12_to_bytes(&halo_square),
                "square mismatch"
            );

            let ark_mul = ark_lhs * ark_rhs;
            let halo_mul = halo_lhs * halo_rhs;

            assert_eq!(
                fq12_to_bytes(&ark_mul).to_vec(),
                halo_fq12_to_bytes(&halo_mul),
                "mul mismatch"
            );
        }
    }

    #[test]
    fn test_fq12_to_bytes_matches_halo2_layout() {
        let mut rng = StdRng::seed_from_u64(24);
        for _ in 0..5 {
            let value = Fq12::rand(&mut rng);
            let halo = ark_fq12_to_halo(&value);
            let expected: Vec<u8> = halo
                .to_coeffs()
                .into_iter()
                .flat_map(|fp2| fp2.to_coeffs())
                .flat_map(|fp| fp.to_bytes())
                .collect();
            assert_eq!(fq12_to_bytes(&value), expected.as_slice());
        }
    }

    #[test]
    fn test_miller_loop_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..5 {
            // Step 1: Create random ark points
            let ark_g1 = G1Affine::rand(&mut rng);
            let ark_g2 = G2Affine::rand(&mut rng);

            let halo_g1_x = HaloFq::from_bytes(&fq_to_bytes(&ark_g1.x)).unwrap();
            let halo_g1_y = HaloFq::from_bytes(&fq_to_bytes(&ark_g1.y)).unwrap();
            let halo_g2_x = halo_fq2_from_bytes(&fq2_to_bytes(&ark_g2.x));
            let halo_g2_y = halo_fq2_from_bytes(&fq2_to_bytes(&ark_g2.y));

            let halo_ps = vec![AffinePoint::new(halo_g1_x, halo_g1_y)];
            let halo_qs = vec![AffinePoint::new(halo_g2_x, halo_g2_y)];

            let halo_result = GuestBn254::multi_miller_loop(&halo_ps, &halo_qs);
            let ark_result = super::multi_miller_loop(&[ark_g1], &[ark_g2]);

            assert_eq!(
                fq12_to_bytes(&ark_result).to_vec(),
                halo_fq12_to_bytes(&halo_result),
                "miller mismatch"
            );
        }
    }

    #[test]
    fn test_final_exp_hint_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..5 {
            let ark_g1 = G1Affine::rand(&mut rng);
            let ark_g2 = G2Affine::rand(&mut rng);

            let halo_g1_x = HaloFq::from_bytes(&fq_to_bytes(&ark_g1.x)).unwrap();
            let halo_g1_y = HaloFq::from_bytes(&fq_to_bytes(&ark_g1.y)).unwrap();
            let halo_g2_x = halo_fq2_from_bytes(&fq2_to_bytes(&ark_g2.x));
            let halo_g2_y = halo_fq2_from_bytes(&fq2_to_bytes(&ark_g2.y));

            let halo_ps = vec![AffinePoint::new(halo_g1_x, halo_g1_y)];
            let halo_qs = vec![AffinePoint::new(halo_g2_x, halo_g2_y)];

            let halo_miller = GuestBn254::multi_miller_loop(&halo_ps, &halo_qs);
            let ark_miller = super::multi_miller_loop(&[ark_g1], &[ark_g2]);

            let (c_halo, u_halo) = GuestBn254::final_exp_hint(&halo_miller);
            let (c_ark, u_ark) = super::final_exp_witness(&ark_miller);

            assert_eq!(
                fq12_to_bytes(&c_ark).to_vec(),
                halo_fq12_to_bytes(&c_halo),
                "c mismatch"
            );
            assert_eq!(
                fq12_to_bytes(&u_ark).to_vec(),
                halo_fq12_to_bytes(&u_halo),
                "u mismatch"
            );
        }
    }
}
