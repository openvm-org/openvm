extern crate alloc;

#[allow(unused_imports)]
use openvm_algebra_guest::{DivUnsafe, IntMod};
use openvm_k256::{Secp256k1Coord, Secp256k1Scalar};
use openvm_p256::{P256Coord, P256Scalar};
use openvm_pairing::{bls12_381::Bls12_381Fp, bn254::Bn254Fp};

openvm::init!();

// Based on https://en.wikipedia.org/wiki/Fermat%27s_little_theorem. If this
// fails, then F::MODULUS is not prime.
fn fermat<F: IntMod>()
where
    F::Repr: AsRef<[u8]>,
{
    let mut pow = F::MODULUS;
    pow.as_mut()[0] -= 2;

    let mut a = F::from_u32(1234);
    let mut res = F::from_u32(1);
    let inv = res.clone().div_unsafe(&a);

    for pow_bit in pow.as_ref() {
        for j in 0..8 {
            if pow_bit & (1 << j) != 0 {
                res *= &a;
            }
            a *= a.clone();
        }
    }

    assert_eq!(res, inv);
}

pub fn main() {
    fermat::<Bn254Fp>();
    fermat::<Bls12_381Fp>();
    fermat::<Secp256k1Coord>();
    fermat::<Secp256k1Scalar>();
    fermat::<P256Coord>();
    fermat::<P256Scalar>();
}
