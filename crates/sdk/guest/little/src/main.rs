extern crate alloc;

use openvm_algebra_guest::IntMod;
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

    let a = F::from_u32(1234);
    let mut res = F::ONE;
    let mut mut_a = a.clone();

    for pow_byte in pow.as_ref() {
        for j in 0..8 {
            if pow_byte & (1 << j) != 0 {
                res *= &mut_a;
            }
            mut_a *= mut_a.clone();
        }
    }

    assert_eq!(res * a, F::ONE);
}

pub fn main() {
    fermat::<Bn254Fp>();
    fermat::<Bls12_381Fp>();
    fermat::<Secp256k1Coord>();
    fermat::<Secp256k1Scalar>();
    fermat::<P256Coord>();
    fermat::<P256Scalar>();
}
