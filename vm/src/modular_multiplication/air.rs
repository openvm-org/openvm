use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;

use crate::cpu::{OpCode, MODULAR_ARITHMETIC_INSTRUCTIONS};

pub struct ModularArithmeticVmAir {
    pub air: ModularArithmeticBigIntAir,
}

impl ModularArithmeticVmAir {
    pub(crate) fn max_accesses_per_instruction(op_code: OpCode) -> usize {
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&op_code));
        1000
    }
}
