use ark_bls12_381::{Fq, Fq12, Fq2, Fq6, G1Affine, G2Affine};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::{Num, Zero};
use std::convert::TryInto;

use super::{FQ12_NUM_BYTES, FQ2_NUM_BYTES, FQ6_NUM_BYTES, FQ_NUM_BYTES};

lazy_static! {
    static ref POLY_FACTOR: BigUint =
        BigUint::from_str_radix("5044125407647214251", 10).unwrap();
    static ref FINAL_EXP_FACTOR: BigUint = BigUint::from_str_radix("2366356426548243601069753987687709088104621721678962410379583120840019275952471579477684846670499039076873213559162845121989217658133790336552276567078487633052653005423051750848782286407340332979263075575489766963251914185767058009683318020965829271737924625612375201545022326908440428522712877494557944965298566001441468676802477524234094954960009227631543471415676620753242466901942121887152806837594306028649150255258504417829961387165043999299071444887652375514277477719817175923289019181393803729926249507024121957184340179467502106891835144220611408665090353102353194448552304429530104218473070114105759487413726485729058069746063140422361472585604626055492939586602274983146215294625774144156395553405525711143696689756441298365274341189385646499074862712688473936093315628166094221735056483459332831845007196600723053356837526749543765815988577005929923802636375670820616189737737304893769679803809426304143627363860243558537831172903494450556755190448279875942974830469855835666815454271389438587399739607656399812689280234103023464545891697941661992848552456326290792224091557256350095392859243101357349751064730561345062266850238821755009430903520645523345000326783803935359711318798844368754833295302563158150573540616830138810935344206231367357992991289265295323280", 10).unwrap();
    static ref FINAL_EXP_FACTOR_TIMES_27: BigUint = FINAL_EXP_FACTOR.clone() * BigUint::from(27u32);
    static ref FINAL_EXP_FACTOR_TIMES_27_BE: Vec<u8> = FINAL_EXP_FACTOR_TIMES_27.to_bytes_be();
    static ref PTH_ROOT_INV_EXP_BE: Vec<u8> = {
        let exp_inv = FINAL_EXP_FACTOR_TIMES_27
            .modinv(&POLY_FACTOR.clone())
            .unwrap();
        let exp = neg_mod(&exp_inv, &POLY_FACTOR);
        exp.to_bytes_be()
    };
    static ref POLY_FACTOR_TIMES_FINAL_EXP: BigUint = POLY_FACTOR.clone() * FINAL_EXP_FACTOR.clone();
    static ref POLY_FACTOR_TIMES_FINAL_EXP_BE: Vec<u8> = POLY_FACTOR_TIMES_FINAL_EXP.to_bytes_be();
    static ref THREE: BigUint = BigUint::from(3u32);
    static ref THREE_BE: Vec<u8> = THREE.to_bytes_be();
    static ref ROOT27_EXPONENT_BYTES: [Vec<u8>; 3] = {
        let exponent = POLY_FACTOR_TIMES_FINAL_EXP.clone();
        let moduli = [THREE.clone(), THREE.clone().pow(2u32), THREE.clone().pow(3u32)];
        let mut exps = Vec::with_capacity(3);
        for modulus in moduli.iter() {
            let inv = exponent.modinv(modulus).unwrap();
            let exp = neg_mod(&inv, modulus);
            exps.push(exp.to_bytes_be());
        }
        exps.try_into().expect("three exponents")
    };
    static ref LAMBDA: BigUint = BigUint::from_str_radix(
        "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129030796414117214202539",
        10
    )
    .unwrap();
    static ref LAMBDA_INV_MOD_FINAL_EXP_BE: Vec<u8> = {
        let exponent = LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
        exponent.to_bytes_be()
    };
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
    let (c, s) = final_exp_witness(&miller);
    // Emit bytes in the Halo layout so guest code can consume the hint directly.
    let mut bytes = Vec::with_capacity(2 * FQ12_NUM_BYTES);
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&c));
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&s));
    bytes
}

fn multi_miller_loop(p: &[G1Affine], q: &[G2Affine]) -> Fq12 {
    // DO NOT USE THESE. THEY DON'T WORK
    // let g1_it = p.iter().map(|point| prepare_g1::<ArkBls12_381>(*point));
    // let g2_it = q.iter().map(|point| prepare_g2::<ArkBls12_381>(*point));
    // ArkBls12_381::multi_miller_loop(g1_it, g2_it).0
    crate::arkworks::miller_loop_bls12_381::multi_miller_loop(p, q)
}

fn final_exp_witness(f: &Fq12) -> (Fq12, Fq12) {
    #[cfg(not(target_os = "zkvm"))]
    println!("[arkworks bls12 final_exp_witness] start f={:?}", f);

    let root = exp_bytes(f, true, FINAL_EXP_FACTOR_TIMES_27_BE.as_slice());
    #[cfg(not(target_os = "zkvm"))]
    println!("[arkworks bls12 final_exp_witness] root={:?}", root);
    let root_pth_inverse = if root == Fq12::ONE {
        Fq12::ONE
    } else {
        exp_bytes(&root, true, PTH_ROOT_INV_EXP_BE.as_slice())
    };
    #[cfg(not(target_os = "zkvm"))]
    println!(
        "[arkworks bls12 final_exp_witness] root_pth_inverse={:?}",
        root_pth_inverse
    );

    let mut order_3rd_power = 0u32;
    let mut root_order = exp_bytes(f, true, POLY_FACTOR_TIMES_FINAL_EXP_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 0;
    }
    root_order = exp_bytes(&root_order, true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 1;
    }
    root_order = exp_bytes(&root_order, true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 2;
    }
    root_order = exp_bytes(&root_order, true, THREE_BE.as_slice());
    if root_order == Fq12::ONE {
        order_3rd_power = 3;
    }
    #[cfg(not(target_os = "zkvm"))]
    println!(
        "[arkworks bls12 final_exp_witness] order_3rd_power={}",
        order_3rd_power
    );

    let root_27th_inverse = if order_3rd_power == 0 {
        Fq12::ONE
    } else {
        let exponent_bytes = &ROOT27_EXPONENT_BYTES[(order_3rd_power - 1) as usize];
        let root = exp_bytes(f, true, POLY_FACTOR_TIMES_FINAL_EXP_BE.as_slice());
        exp_bytes(&root, true, exponent_bytes.as_slice())
    };
    #[cfg(not(target_os = "zkvm"))]
    println!(
        "[arkworks bls12 final_exp_witness] root_27th_inverse={:?}",
        root_27th_inverse
    );

    let scaling_factor = root_pth_inverse * root_27th_inverse;
    let shifted = *f * scaling_factor;
    let residue_witness = exp_bytes(&shifted, true, LAMBDA_INV_MOD_FINAL_EXP_BE.as_slice());
    #[cfg(not(target_os = "zkvm"))]
    println!(
        "[arkworks bls12 final_exp_witness] scaling_factor={:?}, residue_witness={:?}",
        scaling_factor, residue_witness
    );

    (residue_witness, scaling_factor)
}

fn exp_bytes(base: &Fq12, is_positive: bool, bytes_be: &[u8]) -> Fq12 {
    use num_traits::Zero;

    let mut element = *base;
    if !is_positive {
        element = element.inverse().expect("attempted to invert zero element");
    }

    let exp = BigUint::from_bytes_be(bytes_be);
    if exp.is_zero() {
        return Fq12::ONE;
    }
    let limbs = exp.to_u64_digits();
    element.pow(limbs.as_slice())
}

fn fq12_to_bytes(value: &Fq12) -> [u8; FQ12_NUM_BYTES] {
    let mut bytes = [0u8; FQ12_NUM_BYTES];
    value
        .serialize_uncompressed(&mut bytes[..])
        .expect("Failed to serialize Fq12");
    bytes
}

fn fq12_to_bytes_halo_layout(value: &Fq12) -> [u8; FQ12_NUM_BYTES] {
    // Halo encodes coefficients as [c0.c0, c1.c0, c0.c1, c1.c1, c0.c2, c1.c2].
    let coeffs = [
        value.c0.c0,
        value.c1.c0,
        value.c0.c1,
        value.c1.c1,
        value.c0.c2,
        value.c1.c2,
    ];

    let mut bytes = [0u8; FQ12_NUM_BYTES];
    let mut offset = 0;
    for fq2 in coeffs {
        for limb in [fq2.c0, fq2.c1] {
            let limb_bytes = fq_to_bytes(&limb);
            bytes[offset..offset + FQ_NUM_BYTES].copy_from_slice(&limb_bytes);
            offset += FQ_NUM_BYTES;
        }
    }
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

fn neg_mod(value: &BigUint, modulus: &BigUint) -> BigUint {
    let value_mod = value % modulus;
    if value_mod.is_zero() {
        BigUint::ZERO
    } else {
        modulus - value_mod
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bls12_381::halo2curves as bls12_halo;
    use ark_bls12_381::Fq6;
    use ark_ec::AdditiveGroup;
    use ark_ff::{Field as ArkField, UniformRand};
    use halo2curves_axiom::bls12_381::{
        Fq as HaloFq, Fq12 as HaloFq12, Fq2 as HaloFq2, Fq6 as HaloFq6, G1Affine as HaloG1,
        G2Affine as HaloG2, MillerLoopResult,
    };
    use halo2curves_axiom::ff::Field as HaloField;
    use openvm_ecc_guest::algebra::field::FieldExtension;
    use openvm_ecc_guest::AffinePoint;
    use openvm_pairing_guest::halo2curves_shims::bls12_381::test_utils::{
        assert_miller_results_eq, final_exp,
    };
    use openvm_pairing_guest::{
        halo2curves_shims::bls12_381::Bls12_381 as GuestBls12_381,
        pairing::{FinalExp, MultiMillerLoop},
    };
    use rand::{rngs::StdRng, SeedableRng};
    use std::convert::TryInto;

    fn ark_fq_to_halo(value: &Fq) -> HaloFq {
        HaloFq::from_bytes(&fq_to_bytes(value)).unwrap()
    }

    fn ark_fq2_to_halo(value: &Fq2) -> HaloFq2 {
        HaloFq2 {
            c0: ark_fq_to_halo(&value.c0),
            c1: ark_fq_to_halo(&value.c1),
        }
    }

    fn halo_fq_to_ark(value: &HaloFq) -> Fq {
        let bytes = value.to_bytes();
        Fq::deserialize_uncompressed(&bytes[..]).unwrap()
    }

    fn halo_fq2_to_ark(value: &HaloFq2) -> Fq2 {
        Fq2::new(halo_fq_to_ark(&value.c0), halo_fq_to_ark(&value.c1))
    }

    fn ark_fq12_to_halo(value: &Fq12) -> HaloFq12 {
        HaloFq12::from_coeffs([
            ark_fq2_to_halo(&value.c0.c0),
            ark_fq2_to_halo(&value.c1.c0),
            ark_fq2_to_halo(&value.c0.c1),
            ark_fq2_to_halo(&value.c1.c1),
            ark_fq2_to_halo(&value.c0.c2),
            ark_fq2_to_halo(&value.c1.c2),
        ])
    }

    fn halo_fq12_to_ark(value: &HaloFq12) -> Fq12 {
        let [c0_c0, c1_c0, c0_c1, c1_c1, c0_c2, c1_c2] = value.to_coeffs();
        Fq12::new(
            Fq6::new(
                halo_fq2_to_ark(&c0_c0),
                halo_fq2_to_ark(&c0_c1),
                halo_fq2_to_ark(&c0_c2),
            ),
            Fq6::new(
                halo_fq2_to_ark(&c1_c0),
                halo_fq2_to_ark(&c1_c1),
                halo_fq2_to_ark(&c1_c2),
            ),
        )
    }

    fn halo_g1_to_ark(g: &HaloG1) -> G1Affine {
        G1Affine::new_unchecked(halo_fq_to_ark(&g.x), halo_fq_to_ark(&g.y))
    }

    fn halo_g2_to_ark(g: &HaloG2) -> G2Affine {
        G2Affine::new_unchecked(halo_fq2_to_ark(&g.x), halo_fq2_to_ark(&g.y))
    }

    fn halo_g1_to_affine_point(g: &HaloG1) -> AffinePoint<HaloFq> {
        AffinePoint::new(g.x.clone(), g.y.clone())
    }

    fn halo_g2_to_affine_point(g: &HaloG2) -> AffinePoint<HaloFq2> {
        AffinePoint::new(g.x.clone(), g.y.clone())
    }

    fn halo_fq12_to_bytes(value: &HaloFq12) -> Vec<u8> {
        let [c0_c0, c1_c0, c0_c1, c1_c1, c0_c2, c1_c2] = value.to_coeffs();
        [c0_c0, c0_c1, c0_c2, c1_c0, c1_c1, c1_c2]
            .into_iter()
            .flat_map(|fq2| fq2.to_coeffs())
            .flat_map(|fq| fq.to_bytes())
            .collect()
    }

    fn ark_fq12_from_bytes(bytes: &[u8]) -> Fq12 {
        assert_eq!(bytes.len(), FQ12_NUM_BYTES);
        let mut fq2_coeffs = Vec::with_capacity(6);
        let mut offset = 0;
        for _ in 0..6 {
            let c0 = Fq::deserialize_uncompressed(&bytes[offset..offset + FQ_NUM_BYTES]).unwrap();
            offset += FQ_NUM_BYTES;
            let c1 = Fq::deserialize_uncompressed(&bytes[offset..offset + FQ_NUM_BYTES]).unwrap();
            offset += FQ_NUM_BYTES;
            fq2_coeffs.push(Fq2::new(c0, c1));
        }
        Fq12::new(
            Fq6::new(fq2_coeffs[0], fq2_coeffs[1], fq2_coeffs[2]),
            Fq6::new(fq2_coeffs[3], fq2_coeffs[4], fq2_coeffs[5]),
        )
    }

    fn halo_fq12_from_bytes(bytes: &[u8]) -> HaloFq12 {
        assert_eq!(bytes.len(), FQ12_NUM_BYTES);
        let mut fq2_coeffs = Vec::with_capacity(6);
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
            fq2_coeffs.push(HaloFq2 { c0, c1 });
        }
        let fq2_coeffs: [HaloFq2; 6] = fq2_coeffs.try_into().unwrap();
        let [c0_c0, c0_c1, c0_c2, c1_c0, c1_c1, c1_c2] = fq2_coeffs;
        HaloFq12::from_coeffs([c0_c0, c1_c0, c0_c1, c1_c1, c0_c2, c1_c2])
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
        HaloFq2 { c0, c1 }
    }

    fn halo_fq2_to_bytes(value: &HaloFq2) -> Vec<u8> {
        value
            .to_coeffs()
            .into_iter()
            .flat_map(|fq| fq.to_bytes())
            .collect()
    }

    fn ark_fq6_from_bytes(bytes: &[u8]) -> Fq6 {
        assert_eq!(bytes.len(), FQ6_NUM_BYTES);
        Fq6::deserialize_uncompressed(bytes).unwrap()
    }

    fn halo_fq6_from_bytes(bytes: &[u8]) -> HaloFq6 {
        assert_eq!(bytes.len(), FQ6_NUM_BYTES);
        let mut fq2_coeffs = Vec::with_capacity(3);
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
            fq2_coeffs.push(HaloFq2 { c0, c1 });
        }
        HaloFq6 {
            c0: fq2_coeffs[0],
            c1: fq2_coeffs[1],
            c2: fq2_coeffs[2],
        }
    }

    fn halo_fq6_to_bytes(value: &HaloFq6) -> Vec<u8> {
        [value.c0, value.c1, value.c2]
            .into_iter()
            .flat_map(|fq2| fq2.to_coeffs())
            .flat_map(|fq| fq.to_bytes())
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
            assert_eq!(fq12_to_bytes(&value).to_vec(), halo_fq12_to_bytes(&halo));
        }
    }

    #[test]
    fn test_final_exp_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..5 {
            let halo_g1 = HaloG1::random(&mut rng);
            let halo_g2 = HaloG2::random(&mut rng);

            let ark_g1 = halo_g1_to_ark(&halo_g1);
            let ark_g2 = halo_g2_to_ark(&halo_g2);

            let halo_ps = vec![AffinePoint::new(halo_g1.x, halo_g1.y)];
            let halo_qs = vec![AffinePoint::new(halo_g2.x, halo_g2.y)];

            let halo_miller = GuestBls12_381::multi_miller_loop(&halo_ps, &halo_qs);
            let halo_exp = final_exp(halo_miller);

            let ark_miller = super::multi_miller_loop(&[ark_g1], &[ark_g2]);
            let ark_exp = final_exp(ark_fq12_to_halo(&ark_miller));
            assert_eq!(halo_exp, ark_exp, "ark exponentiation mismatch");

            // let ark_miller_output = {
            //     let g1_it = std::iter::once(ark_g1).map(|point| prepare_g1::<ArkBls12_381>(point));
            //     let g2_it = std::iter::once(ark_g2).map(|point| prepare_g2::<ArkBls12_381>(point));
            //     ArkBls12_381::multi_miller_loop(g1_it, g2_it)
            // };
            // let ark_exp_output = ArkBls12_381::final_exponentiation(ark_miller_output)
            //     .unwrap()
            //     .0;
            // assert_eq!(
            //     halo_exp,
            //     ark_fq12_to_halo(&ark_exp_output),
            //     "ark exponentiation output mismatch"
            // );
        }
    }

    #[test]
    fn test_final_exp_hint_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..5 {
            let halo_g1 = HaloG1::random(&mut rng);
            let halo_g2 = HaloG2::random(&mut rng);

            let ark_g1 = halo_g1_to_ark(&halo_g1);
            let ark_g2 = halo_g2_to_ark(&halo_g2);

            let halo_ps = vec![AffinePoint::new(halo_g1.x, halo_g1.y)];
            let halo_qs = vec![AffinePoint::new(halo_g2.x, halo_g2.y)];

            let halo_miller = GuestBls12_381::multi_miller_loop(&halo_ps, &halo_qs);
            let halo_exp = final_exp(halo_miller);

            let ark_miller = super::multi_miller_loop(&[ark_g1], &[ark_g2]);
            let ark_exp = final_exp(ark_fq12_to_halo(&ark_miller));
            assert_eq!(halo_exp, ark_exp, "ark exponentiation mismatch");

            // let ark_miller_output = MillerLoopOutput::<ArkBls12_381>(ark_miller);
            // let ark_exp_output = ArkBls12_381::final_exponentiation(ark_miller_output)
            //     .unwrap()
            //     .0;
            // assert_eq!(
            //     halo_exp,
            //     ark_fq12_to_halo(&ark_exp_output),
            //     "ark exponentiation output mismatch"
            // );

            let (c_halo, u_halo) = GuestBls12_381::final_exp_hint(&halo_miller);
            let (c_ark, u_ark) = super::final_exp_witness(&halo_fq12_to_ark(&halo_miller));
            // let (c_ark, u_ark) = super::final_exp_witness(&ark_miller);

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

    #[test]
    fn test_pairing_hint_bytes_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..5 {
            let ark_g1 = G1Affine::rand(&mut rng);
            let ark_g2 = G2Affine::rand(&mut rng);

            let raw_p = vec![[fq_to_bytes(&ark_g1.x), fq_to_bytes(&ark_g1.y)]];
            let raw_q = vec![[
                [fq_to_bytes(&ark_g2.x.c0), fq_to_bytes(&ark_g2.x.c1)],
                [fq_to_bytes(&ark_g2.y.c0), fq_to_bytes(&ark_g2.y.c1)],
            ]];

            let p_ark = super::parse_g1_points(raw_p.clone());
            let q_ark = super::parse_g2_points(raw_q.clone());
            let bytes_ark = super::pairing_hint_bytes(&p_ark, &q_ark);

            let p_halo = bls12_halo::parse_g1_points(raw_p);
            let q_halo = bls12_halo::parse_g2_points(raw_q);
            let bytes_halo = bls12_halo::pairing_hint_bytes(&p_halo, &q_halo);

            assert_eq!(bytes_ark, bytes_halo, "pairing hint bytes mismatch");
        }
    }
}
