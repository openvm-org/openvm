#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]
#![allow(unused_imports)]

extern crate alloc;

use openvm::io::read_vec;
use openvm_algebra_guest::{
    field::{ComplexConjugate, FieldExtension},
    DivUnsafe, Field, IntMod,
};
use openvm_ecc_guest::AffinePoint;
use openvm_pairing::PairingCheck;
use openvm_pairing_guest::pairing::{exp_check_fallback, MultiMillerLoop, PairingCheckError};

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use alloc::format;

    use openvm_pairing::bn254::{Bn254, Fp, Fp12, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_check_fallback_bn254.rs");

    // Wrapper so that we can override `pairing_check_hint`
    struct Bn254Wrapper(Bn254);

    #[allow(non_snake_case)]
    impl PairingCheck for Bn254Wrapper {
        type Fp = Fp;
        type Fp2 = Fp2;
        type Fp12 = Fp12;

        fn pairing_check_hint(
            _P: &[AffinePoint<Self::Fp>],
            _Q: &[AffinePoint<Self::Fp2>],
        ) -> (Self::Fp12, Self::Fp12) {
            // return dummy values
            (Fp12::ZERO, Fp12::ZERO)
        }

        // copied from Bn254::pairing_check
        fn pairing_check(
            P: &[AffinePoint<Self::Fp>],
            Q: &[AffinePoint<Self::Fp2>],
        ) -> Result<(), PairingCheckError> {
            Self::try_honest_pairing_check(P, Q).unwrap_or_else(|| {
                let f = Bn254::multi_miller_loop(P, Q);
                exp_check_fallback(&f, &Bn254::FINAL_EXPONENT)
            })
        }
    }

    #[allow(non_snake_case)]
    impl Bn254Wrapper {
        // copied from Bn254::try_honest_pairing_check
        fn try_honest_pairing_check(
            P: &[AffinePoint<<Self as PairingCheck>::Fp>],
            Q: &[AffinePoint<<Self as PairingCheck>::Fp2>],
        ) -> Option<Result<(), PairingCheckError>> {
            let (c, u) = Self::pairing_check_hint(P, Q);
            if c == Fp12::ZERO {
                return None;
            }
            // Hint is only honest if `u` lies in proper subfield Fp6. Matches <https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bn254/e12_pairing.go#L450>
            // The Fp6 representation is `fp6_c0 = [s.c[0], s.c[2], s.c[4]]` and `fp6_c1 = [s.c[1],
            // s.c[3], s.c[5]]`.
            for i in [1, 3, 5] {
                if u.c[i] != Fp2::ZERO {
                    return None;
                }
            }
            let c_inv = Fp12::ONE.div_unsafe(&c);

            // We follow Theorem 3 of https://eprint.iacr.org/2024/640.pdf to check that the pairing equals 1
            // By the theorem, it suffices to provide c and u such that f * u == c^λ.
            // Since λ = 6x + 2 + q^3 - q^2 + q, we will check the equivalent condition:
            // f * c^-{6x + 2} * u * c^-{q^3 - q^2 + q} == 1
            // This is because we can compute f * c^-{6x+2} by embedding the c^-{6x+2} computation
            // in the miller loop.

            // c_mul = c^-{q^3 - q^2 + q}
            let c_q3_inv = FieldExtension::frobenius_map(&c_inv, 3);
            let c_q2 = FieldExtension::frobenius_map(&c, 2);
            let c_q_inv = FieldExtension::frobenius_map(&c_inv, 1);
            let c_mul = c_q3_inv * c_q2 * c_q_inv;

            // Pass c inverse into the miller loop so that we compute fc == f * c^-{6x + 2}
            let fc = Bn254::multi_miller_loop_embedded_exp(P, Q, Some(c_inv));

            if fc * c_mul * u == Fp12::ONE {
                Some(Ok(()))
            } else {
                None
            }
        }
    }

    pub fn test_pairing_check(io: &[u8]) {
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];

        let s0_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s0[..32]),
            Fp::from_le_bytes_unchecked(&s0[32..64]),
        );
        let s1_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s1[..32]),
            Fp::from_le_bytes_unchecked(&s1[32..64]),
        );
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..64]), Fp2::from_bytes(&q0[64..128]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..64]), Fp2::from_bytes(&q1[64..128]));

        let f = Bn254Wrapper::pairing_check(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Ok(()));

        let f = Bn254Wrapper::pairing_check(
            &[-s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Err(PairingCheckError));
    }
}

#[cfg(feature = "bls12_381")]
mod bls12_381 {

    use alloc::format;

    use openvm_pairing::bls12_381::{Bls12_381, Fp, Fp12, Fp2};

    use super::*;

    openvm::init!("openvm_init_pairing_check_fallback_bls12_381.rs");

    // Wrapper so that we can override `pairing_check_hint`
    struct Bls12_381Wrapper(Bls12_381);

    #[allow(non_snake_case)]
    impl PairingCheck for Bls12_381Wrapper {
        type Fp = Fp;
        type Fp2 = Fp2;
        type Fp12 = Fp12;

        #[allow(unused_variables)]
        fn pairing_check_hint(
            _P: &[AffinePoint<Self::Fp>],
            _Q: &[AffinePoint<Self::Fp2>],
        ) -> (Self::Fp12, Self::Fp12) {
            // return dummy values
            (Fp12::ZERO, Fp12::ZERO)
        }

        // copied from Bls12_381::pairing_check
        fn pairing_check(
            P: &[AffinePoint<Self::Fp>],
            Q: &[AffinePoint<Self::Fp2>],
        ) -> Result<(), PairingCheckError> {
            Self::try_honest_pairing_check(P, Q).unwrap_or_else(|| {
                let f = Bls12_381::multi_miller_loop(P, Q);
                exp_check_fallback(&f, &Bls12_381::FINAL_EXPONENT)
            })
        }
    }

    #[allow(non_snake_case)]
    impl Bls12_381Wrapper {
        // copied from Bls12_381::try_honest_pairing_check
        fn try_honest_pairing_check(
            P: &[AffinePoint<<Self as PairingCheck>::Fp>],
            Q: &[AffinePoint<<Self as PairingCheck>::Fp2>],
        ) -> Option<Result<(), PairingCheckError>> {
            let (c, s) = Self::pairing_check_hint(P, Q);
            // Hint is only honest if `s` lies in proper subfield Fp6. Matches <https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/fields_bls12381/e12_pairing.go#L413>
            // The Fp6 representation is `fp6_c0 = [s.c[0], s.c[2], s.c[4]]` and `fp6_c1 = [s.c[1],
            // s.c[3], s.c[5]]`.
            for i in [1, 3, 5] {
                if s.c[i] != Fp2::ZERO {
                    return None;
                }
            }

            // The gnark implementation checks that f * s = c^{q - x} where x is the curve seed.
            // We check an equivalent condition: f * c^x * s = c^q.
            // This is because we can compute f * c^x by embedding the c^x computation in the miller
            // loop.

            // We compute c^q before c is consumed by conjugate() below
            let c_q = FieldExtension::frobenius_map(&c, 1);

            // Since the Bls12_381 curve has a negative seed, the miller loop for Bls12_381 is
            // computed as f_{Miller,x,Q}(P) = conjugate( f_{Miller,-x,Q}(P) * c^{-x} ).
            // We will pass in the conjugate inverse of c into the miller loop so that we compute
            // fc = conjugate( f_{Miller,-x,Q}(P) * c'^{-x} )  (where c' is the conjugate inverse of
            // c)    = f_{Miller,x,Q}(P) * c^x
            let c_conj = c.conjugate();
            if c_conj == Fp12::ZERO {
                return None;
            }
            let c_conj_inv = Fp12::ONE.div_unsafe(&c_conj);
            let fc = Bls12_381::multi_miller_loop_embedded_exp(P, Q, Some(c_conj_inv));

            if fc * s == c_q {
                Some(Ok(()))
            } else {
                None
            }
        }
    }

    pub fn test_pairing_check(io: &[u8]) {
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];

        let s0_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s0[..48]),
            Fp::from_le_bytes_unchecked(&s0[48..96]),
        );
        let s1_cast = AffinePoint::new(
            Fp::from_le_bytes_unchecked(&s1[..48]),
            Fp::from_le_bytes_unchecked(&s1[48..96]),
        );
        let q0_cast = AffinePoint::new(Fp2::from_bytes(&q0[..96]), Fp2::from_bytes(&q0[96..192]));
        let q1_cast = AffinePoint::new(Fp2::from_bytes(&q1[..96]), Fp2::from_bytes(&q1[96..192]));

        let f = Bls12_381Wrapper::pairing_check(
            &[s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Ok(()));

        let f = Bls12_381Wrapper::pairing_check(
            &[-s0_cast.clone(), s1_cast.clone()],
            &[q0_cast.clone(), q1_cast.clone()],
        );
        assert_eq!(f, Err(PairingCheckError));
    }
}

pub fn main() {
    #[allow(unused_variables)]
    let io = read_vec();

    #[cfg(feature = "bn254")]
    {
        bn254::test_pairing_check(&io);
    }
    #[cfg(feature = "bls12_381")]
    {
        bls12_381::test_pairing_check(&io);
    }
}
