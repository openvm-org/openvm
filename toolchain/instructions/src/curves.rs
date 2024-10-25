use num_bigint::BigUint;
use num_traits::Num;

use crate::{Fp12Opcode, UsizeOpcode};

const NUM_OPS: usize = 4;

pub trait Curve {
    fn modulus() -> BigUint;
    fn order() -> BigUint;
}

pub struct Bn254Fp12Opcode(Fp12Opcode);

impl Curve for Bn254Fp12Opcode {
    fn modulus() -> BigUint {
        BigUint::from_str_radix(
            "21888242871839275222246405745257275088696311157297823662689037894645226208583",
            10,
        )
        .unwrap()
    }
    fn order() -> BigUint {
        BigUint::from_str_radix(
            "21888242871839275222246405745257275088548364400416034343698204186575808495617",
            10,
        )
        .unwrap()
    }
}

impl UsizeOpcode for Bn254Fp12Opcode {
    fn default_offset() -> usize {
        Fp12Opcode::default_offset()
    }

    fn from_usize(value: usize) -> Self {
        Self(Fp12Opcode::from_usize(value))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize()
    }
}

pub struct Bls12381Fp12Opcode(Fp12Opcode);

impl Curve for Bls12381Fp12Opcode {
    fn modulus() -> BigUint {
        BigUint::from_str_radix(
            "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab",
            16,
        )
        .unwrap()
    }
    fn order() -> BigUint {
        BigUint::from_str_radix(
            "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
            16,
        )
        .unwrap()
    }
}

impl UsizeOpcode for Bls12381Fp12Opcode {
    fn default_offset() -> usize {
        Fp12Opcode::default_offset() + NUM_OPS
    }

    fn from_usize(value: usize) -> Self {
        Self(Fp12Opcode::from_usize(value - NUM_OPS))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize() + NUM_OPS
    }
}
