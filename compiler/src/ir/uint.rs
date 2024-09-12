use p3_field::PrimeField64;
use stark_vm::modular_arithmetic::NUM_LIMBS;

use super::modular_arithmetic::BigUintVar;
use crate::ir::{Builder, Config, DslIr, Ptr};

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn u256_add(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::AddU256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn u256_sub(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::SubU256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn u256_lt(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Ptr<C::N> {
        // let dst = self.alloc(1, <Var<C::N> as MemVariable<C>>::size_of());
        let dst = self.uninit();
        self.operations
            .push(DslIr::LessThanU256(dst, left.clone(), right.clone()));
        dst
    }

    pub fn u256_eq(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Ptr<C::N> {
        // let dst = self.alloc(1, <Var<C::N> as MemVariable<C>>::size_of());
        let dst = self.uninit();
        self.operations
            .push(DslIr::EqualToU256(dst, left.clone(), right.clone()));
        dst
    }
}
