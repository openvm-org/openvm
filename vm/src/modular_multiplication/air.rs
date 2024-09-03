use afs_primitives::modular_multiplication::bigint::air::ModularArithmeticBigIntAir;

use crate::arch::instructions::{
    Opcode, SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS,
    SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS,
};

#[derive(Clone, Debug)]
pub struct ModularArithmeticVmAir {
    pub air: ModularArithmeticBigIntAir,
}

impl ModularArithmeticVmAir {
    pub fn time_stamp_delta(&self) -> usize {
        let num_elems = self.air.limb_dimensions.io_limb_sizes.len();
        3 * (num_elems + 1)
    }
}

impl ModularArithmeticVmAir {
    #[allow(dead_code)]
    pub(crate) fn max_accesses_per_instruction(opcode: Opcode) -> usize {
        assert!(
            SECP256K1_COORD_MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&opcode)
                || SECP256K1_SCALAR_MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&opcode)
        );
        1000
    }
}
