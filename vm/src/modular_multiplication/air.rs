use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;

use crate::cpu::{MODULAR_ARITHMETIC_INSTRUCTIONS, Opcode};

pub struct ModularArithmeticVmAir {
    pub air: ModularArithmeticBigIntAir,
}

<<<<<<< HEAD
impl ModularMultiplicationVmAir {
    pub(crate) fn max_accesses_per_instruction(op_code: Opcode) -> usize {
=======
impl ModularArithmeticVmAir {
    pub(crate) fn max_accesses_per_instruction(op_code: OpCode) -> usize {
>>>>>>> main
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&op_code));
        1000
    }
}
