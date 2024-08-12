use crate::ir::{Builder, Config, modular_arithmetic::BigIntVar};

const EC_CONSTANT: usize = 4;

impl<C: Config> Builder<C> {
    pub fn ec_add(
        &mut self,
        left: (BigIntVar<C>, BigIntVar<C>),
        right: (BigIntVar<C>, BigIntVar<C>),
    ) -> (BigIntVar<C>, BigIntVar<C>) {
        todo!()
    }
}
