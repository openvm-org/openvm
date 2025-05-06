extern crate alloc;

use alloc::vec::Vec;
use core::ops::Neg;

use openvm_algebra_complex_macros::{complex_declare, complex_impl_field};
use openvm_algebra_guest::{field::FieldExtension, Field, IntMod};

use super::Fp;

// The struct name needs to be globally unique for linking purposes.
// The mod_type is a path used only in the struct definition.
complex_declare! {
    Bls12_381Fp2 { mod_type = Fp }
}

complex_impl_field! {
    Bls12_381Fp2,
}

pub type Fp2 = Bls12_381Fp2;

impl FieldExtension<Fp> for Fp2 {
    const D: usize = 2;
    type Coeffs = [Fp; 2];

    fn from_coeffs([c0, c1]: Self::Coeffs) -> Self {
        Self { c0, c1 }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        assert_eq!(bytes.len(), 96);
        Self::from_coeffs([
            Fp::from_const_bytes(bytes[0..48].try_into().unwrap()),
            Fp::from_const_bytes(bytes[48..96].try_into().unwrap()),
        ])
    }

    fn to_coeffs(self) -> Self::Coeffs {
        [self.c0, self.c1]
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(96);
        bytes.extend_from_slice(self.c0.as_le_bytes());
        bytes.extend_from_slice(self.c1.as_le_bytes());
        bytes
    }

    fn embed(base_elem: Fp) -> Self {
        Self {
            c0: base_elem,
            c1: <Fp as Field>::ZERO,
        }
    }

    fn frobenius_map(&self, power: usize) -> Self {
        if power % 2 == 0 {
            self.clone()
        } else {
            Self {
                c0: self.c0.clone(),
                c1: (&self.c1).neg(),
            }
        }
    }

    fn mul_base(&self, rhs: &Fp) -> Self {
        Self {
            c0: &self.c0 * rhs,
            c1: &self.c1 * rhs,
        }
    }
}
