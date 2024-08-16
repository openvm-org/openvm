use afs_primitives::modular_multiplication::bigint::air::ModularMultiplicationBigIntAir;

use crate::cpu::{OpCode, MODULAR_ARITHMETIC_INSTRUCTIONS};

pub struct ModularMultiplicationVmAir {
    pub air: ModularMultiplicationBigIntAir,
}

impl ModularMultiplicationVmAir {
    pub(crate) fn max_accesses_per_instruction(op_code: OpCode) -> usize {
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&op_code));
        1000
    }
}
