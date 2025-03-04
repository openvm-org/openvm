#![feature(cfg_match)]
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
use openvm_pairing_guest::pairing::{
    exp_check_fallback, MultiMillerLoop, PairingCheck, PairingCheckError,
};

openvm::entry!(main);

#[cfg(feature = "bn254")]
mod bn254 {
    use alloc::format;

    use openvm_pairing_guest::bn254::{Bn254, Fp, Fp12, Fp2};

    use super::*;

    openvm_algebra_moduli_macros::moduli_init! {
        "0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
        "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Bn254Fp2 { mod_idx = 0 },
    }

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
            (Fp12::ONE, Fp12::ZERO)
        }

        fn pairing_check(
            P: &[AffinePoint<Self::Fp>],
            Q: &[AffinePoint<Self::Fp2>],
        ) -> Result<(), PairingCheckError> {
            let (c, u) = Self::pairing_check_hint(P, Q);
            // TODO: handle c = 0
            let c_inv = Fp12::ONE.div_unsafe(&c);

            // f * u == c^λ
            // f * u == c^{6x + 2 + q^3 - q^2 + q}
            // f * c^-{6x + 2} * u * c^-{q^3 - q^2 + q} == 1
            // where fc == f * c^-{6x + 2}
            // c_mul = c^-{q^3 - q^2 + q}
            let c_q3_inv = FieldExtension::frobenius_map(&c_inv, 3);
            let c_q2 = FieldExtension::frobenius_map(&c, 2);
            let c_q_inv = FieldExtension::frobenius_map(&c_inv, 1);
            let c_mul = c_q3_inv * c_q2 * c_q_inv;

            // Compute miller loop with c_inv
            let fc = Bn254::multi_miller_loop_embedded_exp(P, Q, Some(c_inv));

            if fc * c_mul * u == Fp12::ONE {
                Ok(())
            } else {
                let f = Bn254::multi_miller_loop(P, Q);
                exp_check_fallback(&f, &Bn254::FINAL_EXPONENT)
            }
        }
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..32 * 2];
        let s1 = &io[32 * 2..32 * 4];
        let q0 = &io[32 * 4..32 * 8];
        let q1 = &io[32 * 8..32 * 12];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

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

    use openvm_pairing_guest::bls12_381::{Bls12_381, Fp, Fp12, Fp2};

    use super::*;

    openvm_algebra_moduli_macros::moduli_init! {
        "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
        "0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"
    }

    openvm_algebra_complex_macros::complex_init! {
        Bls12_381Fp2 { mod_idx = 0 },
    }

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
            (Fp12::ONE, Fp12::ZERO)
        }

        fn pairing_check(
            P: &[AffinePoint<Self::Fp>],
            Q: &[AffinePoint<Self::Fp2>],
        ) -> Result<(), PairingCheckError> {
            let (c, s) = Self::pairing_check_hint(P, Q);

            // f * s = c^{q - x}
            // f * s = c^q * c^-x
            // f * c^x * c^-q * s = 1,
            //   where fc = f * c'^x (embedded Miller loop with c conjugate inverse),
            //   and the curve seed x = -0xd201000000010000
            //   the miller loop computation includes a conjugation at the end because the value of the
            //   seed is negative, so we need to conjugate the miller loop input c as c'. We then substitute
            //   y = -x to get c^-y and finally compute c'^-y as input to the miller loop:
            // f * c'^-y * c^-q * s = 1
            let c_q = FieldExtension::frobenius_map(&c, 1);
            // TODO: handle c = 0
            let c_conj_inv = Fp12::ONE.div_unsafe(&c.conjugate());

            // fc = f_{Miller,x,Q}(P) * c^{x}
            // where
            //   fc = conjugate( f_{Miller,-x,Q}(P) * c'^{-x} ), with c' denoting the conjugate of c
            let fc = Bls12_381::multi_miller_loop_embedded_exp(P, Q, Some(c_conj_inv));

            if fc * s == c_q {
                Ok(())
            } else {
                let f = Bls12_381::multi_miller_loop(P, Q);
                exp_check_fallback(&f, &Bls12_381::FINAL_EXPONENT)
            }
        }
    }

    pub fn test_pairing_check(io: &[u8]) {
        setup_0();
        setup_all_complex_extensions();
        let s0 = &io[0..48 * 2];
        let s1 = &io[48 * 2..48 * 4];
        let q0 = &io[48 * 4..48 * 8];
        let q1 = &io[48 * 8..48 * 12];

        let s0_cast = unsafe { &*(s0.as_ptr() as *const AffinePoint<Fp>) };
        let s1_cast = unsafe { &*(s1.as_ptr() as *const AffinePoint<Fp>) };
        let q0_cast = unsafe { &*(q0.as_ptr() as *const AffinePoint<Fp2>) };
        let q1_cast = unsafe { &*(q1.as_ptr() as *const AffinePoint<Fp2>) };

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

    cfg_match! {
        cfg(feature = "bn254") => { bn254::test_pairing_check(&io); }
        cfg(feature = "bls12_381") => { bls12_381::test_pairing_check(&io); }
        _ => { panic!("No curve feature enabled") }
    }
}
