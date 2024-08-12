use afs_primitives::{
    modular_multiplication::modular_multiplication_bigint::air::ModularMultiplicationBigIntAir,
    sub_chip::AirConfig,
};

use crate::{
    cpu::{OpCode, MODULAR_ARITHMETIC_INSTRUCTIONS},
    modular_multiplication::columns::ModularMultiplicationVmCols,
};

pub struct ModularMultiplicationVmAir {
    pub air: ModularMultiplicationBigIntAir,
}

impl ModularMultiplicationVmAir {
    pub(crate) fn max_accesses_per_instruction(op_code: OpCode) -> usize {
        assert!(MODULAR_ARITHMETIC_INSTRUCTIONS.contains(&op_code));
        1000
    }
}

impl AirConfig for ModularMultiplicationVmAir {
    type Cols<T> = ModularMultiplicationVmCols<T>;
}
