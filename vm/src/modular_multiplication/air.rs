use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;

use crate::arch::instructions::{Opcode, MODULAR_ARITHMETIC_INSTRUCTIONS};

pub struct ModularArithmeticVmAir {
    pub air: ModularArithmeticBigIntAir,
}

impl ModularArithmeticVmAir {
    #[allow(dead_code)]
    pub(crate) fn max_accesses_per_instruction(opcode: Opcode) -> usize {
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&opcode));
        1000
    }
}
