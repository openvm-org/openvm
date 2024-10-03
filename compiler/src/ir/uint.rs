use p3_field::PrimeField64;

use super::{modular_arithmetic::BigUintVar, Var, NUM_LIMBS};
use crate::ir::{Builder, Config, DslIr};

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn add_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::Add256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn sub_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::Sub256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn mul_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::Mul256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn sltu_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Var<C::N> {
        let dst = self.array(1);
        self.operations
            .push(DslIr::LessThanU256(dst.ptr(), left.clone(), right.clone()));
        self.get(&dst, 0)
    }

    pub fn eq_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Var<C::N> {
        // let dst = self.alloc(1, <Var<C::N> as MemVariable<C>>::size_of());
        let dst = self.array(1);
        self.operations
            .push(DslIr::EqualTo256(dst.ptr(), left.clone(), right.clone()));
        self.get(&dst, 0)
    }

    pub fn xor_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::Xor256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn and_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::And256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn or_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(DslIr::Or256(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn slt_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Var<C::N> {
        let dst = self.array(1);
        self.operations
            .push(DslIr::LessThanI256(dst.ptr(), left.clone(), right.clone()));
        self.get(&dst, 0)
    }

    pub fn sll_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations.push(DslIr::ShiftLeft256(
            dst.clone(),
            left.clone(),
            right.clone(),
        ));
        dst
    }

    pub fn srl_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations.push(DslIr::ShiftRightLogic256(
            dst.clone(),
            left.clone(),
            right.clone(),
        ));
        dst
    }

    pub fn sra_256(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations.push(DslIr::ShiftRightArith256(
            dst.clone(),
            left.clone(),
            right.clone(),
        ));
        dst
    }
}
