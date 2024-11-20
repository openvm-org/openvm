use axvm::moduli_setup;
use axvm_algebra::{Field, IntMod};

mod fp12;
mod fp2;
mod fp6;
mod pairing;

pub use fp12::*;
pub use fp2::*;
pub use fp6::*;

use crate::pairing::PairingIntrinsics;

pub struct Bls12_381;

#[cfg(all(test, feature = "halo2curves", not(target_os = "zkvm")))]
mod tests;

moduli_setup! {
    Bls12_381Fp = "0x1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab";
}

pub type Fp = Bls12_381Fp;

impl Field for Fp {
    type SelfRef<'a> = &'a Self;
    const ZERO: Self = <Self as IntMod>::ZERO;
    const ONE: Self = <Self as IntMod>::ONE;

    fn double_assign(&mut self) {
        IntMod::double_assign(self);
    }

    fn square_assign(&mut self) {
        IntMod::square_assign(self);
    }
}

impl PairingIntrinsics for Bls12_381 {
    type Fp = Fp;
    type Fp2 = Fp2;
    type Fp12 = Fp12;

    const PAIRING_IDX: usize = 1;
    const XI: Fp2 = Fp2::new(Fp::from_const_u8(1), Fp::from_const_u8(1));
}

// Inverse z = x⁻¹ (mod p)
pub(crate) fn fp_invert_assign(x: &mut Fp) {
    let res = <Fp as Field>::ONE.div_unsafe_refs_impl(x);
    *x = res;
}

pub(crate) fn fp_square_assign(x: &mut Fp) {
    *x *= x.clone();
}

pub(crate) fn fp_sum_of_products(a: [Fp; 6], b: [Fp; 6]) -> Fp {
    // let (u0, u1, u2, u3, u4, u5) =
    //     (0..6).fold((0, 0, 0, 0, 0, 0), |(u0, u1, u2, u3, u4, u5), j| {
    //         // Algorithm 2, line 3
    //         let (t0, t1, t2, t3, t4, t5, t6) = (0..6).fold(
    //             (u0, u1, u2, u3, u4, u5, 0),
    //             |(t0, t1, t2, t3, t4, t5, t6), i| {
    //                 // Compute digit_j x row and accumulate into `u`
    //                 let (t0, carry) = mac(t0, a[i].0[j], b[i].0[0], 0);
    //                 let (t1, carry) = mac(t1, a[i].0[j], b[i].0[1], carry);
    //                 let (t2, carry) = mac(t2, a[i].0[j], b[i].0[2], carry);
    //                 let (t3, carry) = mac(t3, a[i].0[j], b[i].0[3], carry);
    //                 let (t4, carry) = mac(t4, a[i].0[j], b[i].0[4], carry);
    //                 let (t5, carry) = mac(t5, a[i].0[j], b[i].0[5], carry);
    //                 let (t6, _) = adc(t6, 0, carry);

    //                 (t0, t1, t2, t3, t4, t5, t6)
    //             },
    //         );

    //         // Algorithm 2, lines 4-5
    //         // Single step of Montgomery reduction
    //         let k = t0.wrapping_mul(INV);
    //         let (_, carry) = mac(t0, k, Bls12_381Fp::MODULUS[0..8], 0);
    //         let (r1, carry) = mac(t1, k, Bls12_381Fp::MODULUS[8..16], carry);
    //         let (r2, carry) = mac(t2, k, Bls12_381Fp::MODULUS[16..24], carry);
    //         let (r3, carry) = mac(t3, k, Bls12_381Fp::MODULUS[24..32], carry);
    //         let (r4, carry) = mac(t4, k, Bls12_381Fp::MODULUS[32..40], carry);
    //         let (r5, carry) = mac(t5, k, Bls12_381Fp::MODULUS[40..48], carry);
    //         let (r6, _) = adc(t6, 0, carry);

    //         (r1, r2, r3, r4, r5, r6)
    //     });

    // // Final conditional subtraction for non-redundant form
    // (&Fp([u0, u1, u2, u3, u4, u5])).subtract_p()
    todo!()
}

pub(crate) const fn mac(
    a: [u8; 8],
    b: [u8; 8],
    c: [u8; 8],
    mut carry: [u8; 8],
) -> ([u8; 8], [u8; 8]) {
    let mut result_low = [0u8; 8];
    let mut result_high = [0u8; 8];
    let mut temp_carry: u16 = 0;

    // Process each byte
    let mut i = 0;
    while i < 8 {
        // Multiply b[i] and c[i], add a[i] and previous carry
        let product =
            (b[i] as u16) * (c[i] as u16) + (a[i] as u16) + (carry[i] as u16) + temp_carry;

        // Lower 8 bits go to result
        result_low[i] = product as u8;
        // Upper 8 bits become carry for next iteration
        temp_carry = product >> 8;

        i += 1;
    }

    // Handle final carry
    result_high[0] = temp_carry as u8;

    (result_low, result_high)
}
