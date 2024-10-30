use crate::{EcLineOpcode, Fp12Opcode, UsizeOpcode};

const FP12_OPS: usize = 4;
const ECC_LINE_OPS: usize = 8;

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

pub struct Bn254EcLineOpcode(EcLineOpcode);

impl UsizeOpcode for Bn254EcLineOpcode {
    fn default_offset() -> usize {
        EcLineOpcode::default_offset()
    }

    fn from_usize(value: usize) -> Self {
        Self(EcLineOpcode::from_usize(value))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize()
    }
}

pub struct Bls12381Fp12Opcode(Fp12Opcode);

impl UsizeOpcode for Bls12381Fp12Opcode {
    fn default_offset() -> usize {
        Fp12Opcode::default_offset() + FP12_OPS
    }

    fn from_usize(value: usize) -> Self {
        Self(Fp12Opcode::from_usize(value - FP12_OPS))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize() + FP12_OPS
    }
}

pub struct Bls12381EcLineOpcode(EcLineOpcode);

impl UsizeOpcode for Bls12381EcLineOpcode {
    fn default_offset() -> usize {
        EcLineOpcode::default_offset() + ECC_LINE_OPS
    }

    fn from_usize(value: usize) -> Self {
        Self(EcLineOpcode::from_usize(value - ECC_LINE_OPS))
    }

    fn as_usize(&self) -> usize {
        self.0.as_usize() + ECC_LINE_OPS
    }
}
