use crate::{Fp12Opcode, UsizeOpcode};

const NUM_OPS: usize = 4;

pub struct Bn254Fp12Opcode(Fp12Opcode);

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
