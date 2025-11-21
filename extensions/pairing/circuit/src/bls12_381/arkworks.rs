use ark_bls12_381::{Fq, Fq12, Fq2, G1Affine, G2Affine};
use ark_ff::Field;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use lazy_static::lazy_static;
use num_bigint::BigUint;
use num_traits::{Num, Zero};
use openvm_pairing_guest::halo2curves_shims::naf::biguint_to_naf;

use super::{FQ12_NUM_BYTES, FQ_NUM_BYTES};
use crate::arkworks::exp_naf;

lazy_static! {
    static ref POLY_FACTOR: BigUint =
        BigUint::from_str_radix("5044125407647214251", 10).unwrap();
    static ref FINAL_EXP_FACTOR: BigUint = BigUint::from_str_radix("2366356426548243601069753987687709088104621721678962410379583120840019275952471579477684846670499039076873213559162845121989217658133790336552276567078487633052653005423051750848782286407340332979263075575489766963251914185767058009683318020965829271737924625612375201545022326908440428522712877494557944965298566001441468676802477524234094954960009227631543471415676620753242466901942121887152806837594306028649150255258504417829961387165043999299071444887652375514277477719817175923289019181393803729926249507024121957184340179467502106891835144220611408665090353102353194448552304429530104218473070114105759487413726485729058069746063140422361472585604626055492939586602274983146215294625774144156395553405525711143696689756441298365274341189385646499074862712688473936093315628166094221735056483459332831845007196600723053356837526749543765815988577005929923802636375670820616189737737304893769679803809426304143627363860243558537831172903494450556755190448279875942974830469855835666815454271389438587399739607656399812689280234103023464545891697941661992848552456326290792224091557256350095392859243101357349751064730561345062266850238821755009430903520645523345000326783803935359711318798844368754833295302563158150573540616830138810935344206231367357992991289265295323280", 10).unwrap();
    static ref FINAL_EXP_FACTOR_NAF: Vec<i8> = biguint_to_naf(&FINAL_EXP_FACTOR);
    static ref POLY_FACTOR_NAF: Vec<i8> = biguint_to_naf(&POLY_FACTOR);
    static ref TWENTY_SEVEN_NAF: Vec<i8> = biguint_to_naf(&BigUint::from(27u32));
    static ref TEN_NAF: Vec<i8> = biguint_to_naf(&BigUint::from(10u32));
    static ref FINAL_EXP_TIMES_27: BigUint = FINAL_EXP_FACTOR.clone() * BigUint::from(27u32);
    static ref FINAL_EXP_TIMES_27_MOD_POLY: BigUint = {
        let exp_inv = FINAL_EXP_TIMES_27.modinv(&POLY_FACTOR.clone()).unwrap();
        exp_inv % POLY_FACTOR.clone()
    };
    static ref FINAL_EXP_TIMES_27_MOD_POLY_NAF: Vec<i8> =
        biguint_to_naf(&FINAL_EXP_TIMES_27_MOD_POLY);
    static ref LAMBDA: BigUint = BigUint::from_str_radix(
        "4002409555221667393417789825735904156556882819939007885332058136124031650490837864442687629129030796414117214202539",
        10
    )
    .unwrap();
    static ref LAMBDA_INV_FINAL_EXP: BigUint =
        LAMBDA.clone().modinv(&FINAL_EXP_FACTOR.clone()).unwrap();
    static ref LAMBDA_INV_FINAL_EXP_NAF: Vec<i8> = biguint_to_naf(&LAMBDA_INV_FINAL_EXP);
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
    let (c, s) = final_exp_hint(&miller);
    let mut bytes = Vec::with_capacity(2 * FQ12_NUM_BYTES);
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&c));
    bytes.extend_from_slice(&fq12_to_bytes_halo_layout(&s));
    bytes
}

fn multi_miller_loop_embedded_exp(p: &[G1Affine], q: &[G2Affine], c: Option<Fq12>) -> Fq12 {
    crate::arkworks::miller_loop_bls12_381::multi_miller_loop_embedded_exp(p, q, c)
}

fn multi_miller_loop(p: &[G1Affine], q: &[G2Affine]) -> Fq12 {
    multi_miller_loop_embedded_exp(p, q, None)
}

// Adapted from the gnark implementation:
// https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/hints.go#L273
// returns c (residueWitness) and s (scalingFactor)
// The Gnark implementation is based on https://eprint.iacr.org/2024/640.pdf
fn final_exp_hint(f: &Fq12) -> (Fq12, Fq12) {
    let f_final_exp = exp_naf(f, true, &FINAL_EXP_FACTOR_NAF);
    let root = exp_naf(&f_final_exp, true, &TWENTY_SEVEN_NAF);

    // 1. get p-th root inverse
    let root_pth_inv = if root == Fq12::ONE {
        Fq12::ONE
    } else {
        exp_naf(&root, false, &FINAL_EXP_TIMES_27_MOD_POLY_NAF)
    };

    let root = exp_naf(&f_final_exp, true, &POLY_FACTOR_NAF);
    // 2. get 27th root inverse
    let root_27th_inv = if exp_naf(&root, true, &TWENTY_SEVEN_NAF) == Fq12::ONE {
        exp_naf(&root, false, &TEN_NAF)
    } else {
        Fq12::ONE
    };

    // 2.3. shift the Miller loop result so that millerLoop * scalingFactor
    // is of order finalExpFactor
    let s = root_pth_inv * root_27th_inv;
    let f = *f * s;

    // 3. get the witness residue
    // lambda = q - u, the optimal exponent
    let c = exp_naf(&f, true, &LAMBDA_INV_FINAL_EXP_NAF);

    (c, s)
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

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Fq6, Fr};
    use ark_ec::AffineRepr;
    use ark_ff::{Field as ArkField, UniformRand};
    use halo2curves_axiom::bls12_381::{Fq as HaloFq, Fq12 as HaloFq12, Fq2 as HaloFq2};
    use openvm_ecc_guest::algebra::field::FieldExtension;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::bls12_381::halo2curves as bls12_halo;

    fn fq_to_bytes(value: &Fq) -> [u8; FQ_NUM_BYTES] {
        let mut bytes = [0u8; FQ_NUM_BYTES];
        value
            .serialize_uncompressed(&mut bytes[..])
            .expect("Failed to serialize Fq");
        bytes
    }

    fn halo_fq_to_ark(value: &HaloFq) -> Fq {
        let bytes = value.to_bytes();
        Fq::deserialize_uncompressed(&bytes[..]).unwrap()
    }

    fn halo_fq2_to_ark(value: &HaloFq2) -> Fq2 {
        Fq2::new(halo_fq_to_ark(&value.c0), halo_fq_to_ark(&value.c1))
    }

    fn halo_fq12_to_bytes(value: &HaloFq12) -> Vec<u8> {
        value
            .to_coeffs()
            .into_iter()
            .flat_map(|fq2| fq2.to_coeffs())
            .flat_map(|fq| fq.to_bytes())
            .collect()
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

    fn random_inputs(rng: &mut StdRng) -> (Vec<bls12_halo::G1Affine>, Vec<bls12_halo::G2Affine>) {
        let ark_g1 = G1Affine::rand(rng);
        let ark_g2 = G2Affine::rand(rng);

        let raw_p = vec![[fq_to_bytes(&ark_g1.x), fq_to_bytes(&ark_g1.y)]];
        let raw_q = vec![[
            [fq_to_bytes(&ark_g2.x.c0), fq_to_bytes(&ark_g2.x.c1)],
            [fq_to_bytes(&ark_g2.y.c0), fq_to_bytes(&ark_g2.y.c1)],
        ]];

        (
            bls12_halo::parse_g1_points(raw_p),
            bls12_halo::parse_g2_points(raw_q),
        )
    }

    #[test]
    fn test_final_exp_hint_matches_halo2() {
        let mut rng = StdRng::seed_from_u64(1337);
        for _ in 0..3 {
            let (p_halo, q_halo) = random_inputs(&mut rng);

            let halo_miller = bls12_halo::multi_miller_loop(&p_halo, &q_halo);
            let (c_halo, s_halo) = bls12_halo::final_exp_hint(&halo_miller);
            let (c_ark, s_ark) = final_exp_hint(&halo_fq12_to_ark(&halo_miller));

            assert_eq!(
                fq12_to_bytes_halo_layout(&c_ark).to_vec(),
                halo_fq12_to_bytes(&c_halo),
                "c hint mismatch with halo2"
            );
            assert_eq!(
                fq12_to_bytes_halo_layout(&s_ark).to_vec(),
                halo_fq12_to_bytes(&s_halo),
                "s hint mismatch with halo2"
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

            let p_halo = bls12_halo::parse_g1_points(raw_p);
            let q_halo = bls12_halo::parse_g2_points(raw_q);
            let bytes_halo = bls12_halo::pairing_hint_bytes(&p_halo, &q_halo);

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
        let (c, s) = final_exp_hint(&miller);

        let mut c_q = c;
        c_q.frobenius_map_in_place(1);

        let mut c_conj = c;
        c_conj.conjugate_in_place();
        let c_conj_inv = c_conj.inverse().unwrap();
        let fc = multi_miller_loop_embedded_exp(&p_vec, &q_vec, Some(c_conj_inv));

        assert_eq!(fc * s, c_q, "Pairing check failed");
    }
}
