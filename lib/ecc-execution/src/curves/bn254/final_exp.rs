use axvm_ecc::{
    curve::bn254::{Fq, Fq12, Fq2},
    field::{ExpBigInt, FieldExt},
    pairing::{FinalExp, MultiMillerLoop},
    point::EcPoint,
};
use halo2curves_axiom::ff::Field;

use super::{Bn254, EXP1, EXP2, M_INV, R_INV, U27_COEFF_0, U27_COEFF_1};

#[allow(non_snake_case)]
impl FinalExp<Fq, Fq2, Fq12> for Bn254 {
    fn assert_final_exp_is_one(&self, f: Fq12, P: &[EcPoint<Fq>], Q: &[EcPoint<Fq2>]) {
        let (c, u) = self.final_exp_hint(f);
        let c_inv = c.invert().unwrap();

        // f * u == c^λ
        // f * u == c^{6x + 2 + q^3 - q^2 + q}
        // f * c^-{6x + 2} * u * c^-{q^3 - q^2 + q} == 1
        // where fc == f * c^-{6x + 2}
        // c_mul = c^-{q^3 - q^2 + q}
        let c_q3 = c_inv.frobenius_map(Some(3));
        let c_q2 = c_inv.frobenius_map(Some(2));
        let c_q2_inv = c_q2.invert().unwrap();
        let c_q = c_inv.frobenius_map(Some(1));
        let c_mul = c_q3 * c_q2_inv * c_q;

        // Compute miller loop with c_inv
        let fc = self.multi_miller_loop_embedded_exp(P, Q, Some(c_inv));

        assert_eq!(fc * c_mul * u, Fq12::ONE);
    }

    // Adapted from the Gnark implementation:
    // https://github.com/Consensys/gnark/blob/af754dd1c47a92be375930ae1abfbd134c5310d8/std/algebra/emulated/sw_bn254/hints.go#L23
    // returns c (residueWitness) and u (cubicNonResiduePower)
    fn final_exp_hint(&self, f: Fq12) -> (Fq12, Fq12) {
        // Residue witness
        let mut c;
        // Cubic nonresidue power
        let u;

        // get the 27th root of unity
        let u0 = U27_COEFF_0.to_u64_digits().1;
        let u1 = U27_COEFF_1.to_u64_digits().1;
        let u_coeffs = Fq2::from_coeffs([
            Fq::from_raw([u0[0], u0[1], u0[2], u0[3]]),
            Fq::from_raw([u1[0], u1[1], u1[2], u1[3]]),
        ]);
        let unity_root_27 = Fq12::from_coeffs([
            Fq2::ZERO,
            Fq2::ZERO,
            u_coeffs,
            Fq2::ZERO,
            Fq2::ZERO,
            Fq2::ZERO,
        ]);
        debug_assert_eq!(unity_root_27.pow([27]), Fq12::one());

        if f.exp_bigint(EXP1.clone()) == Fq12::ONE {
            c = f;
            u = Fq12::ONE;
        } else {
            let f_mul_unity_root_27 = f * unity_root_27;
            if f_mul_unity_root_27.exp_bigint(EXP1.clone()) == Fq12::ONE {
                c = f_mul_unity_root_27;
                u = unity_root_27;
            } else {
                c = f_mul_unity_root_27 * unity_root_27;
                u = unity_root_27.square();
            }
        }

        // 1. Compute r-th root and exponentiate to rInv where
        //   rInv = 1/r mod (p^12-1)/r
        c = c.exp_bigint(R_INV.clone());

        // 2. Compute m-th root where
        //   m = (6x + 2 + q^3 - q^2 +q)/3r
        // Exponentiate to mInv where
        //   mInv = 1/m mod p^12-1
        c = c.exp_bigint(M_INV.clone());

        // 3. Compute cube root
        // since gcd(3, (p^12-1)/r) != 1, we use a modified Tonelli-Shanks algorithm
        // see Alg.4 of https://eprint.iacr.org/2024/640.pdf
        // Typo in the paper: p^k-1 = 3^n * s instead of p-1 = 3^r * s
        // where k=12 and n=3 here and exp2 = (s+1)/3
        let mut x = c.exp_bigint(EXP2.clone());

        // 3^t is ord(x^3 / residueWitness)
        let c_inv = c.invert().unwrap();
        let mut x3 = x.square() * x * c_inv;
        let mut t = 0;
        let mut tmp = x3.square();

        // Modified Tonelli-Shanks algorithm for computing the cube root
        fn tonelli_shanks_loop(x3: &mut Fq12, tmp: &mut Fq12, t: &mut i32) {
            while *x3 != Fq12::ONE {
                *tmp = (*x3).square();
                *x3 *= *tmp;
                *t += 1;
            }
        }

        tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);

        while t != 0 {
            tmp = unity_root_27.exp_bigint(EXP2.clone());
            x *= tmp;

            x3 = x.square() * x * c_inv;
            t = 0;
            tonelli_shanks_loop(&mut x3, &mut tmp, &mut t);
        }

        debug_assert_eq!(c, x * x * x);
        // x is the cube root of the residue witness c
        c = x;

        (c, u)
    }
}
