use num_bigint_dig::BigUint;
use p3_field::{AbstractField, PrimeField64};
use stark_vm::modular_multiplication::bigint_to_elems;

use crate::ir::{Array, Builder, Config, DslIr, IfBuilder, Var};

pub type BigIntVar<C> = Array<C, Var<<C as Config>::N>>;

impl<C: Config> BigIntVar<C> {
    pub fn ptr_fp(&self) -> i32 {
        match self {
            Array::Fixed(_) => panic!(),
            Array::Dyn(ptr, _) => ptr.fp(),
        }
    }
}

/// Number of bits of each field element used.
const REPR_BITS: usize = 30;
/// Number of field elements used to represent a bigint.
const NUM_ELEMS: usize = 9;

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn eval_bigint(&mut self, bigint: BigUint) -> BigIntVar<C> {
        let mut array = self.dyn_array(NUM_ELEMS);

        let elems: Vec<C::N> = bigint_to_elems(bigint, REPR_BITS, NUM_ELEMS);
        for (i, &elem) in elems.iter().enumerate() {
            self.set(&mut array, i, elem);
        }

        array
    }

    fn mod_secp256k1_operation(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
        operation: impl Fn(BigIntVar<C>, BigIntVar<C>, BigIntVar<C>) -> DslIr<C>,
    ) -> BigIntVar<C> {
        let dst = self.dyn_array(NUM_ELEMS);
        self.operations
            .push(operation(dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn mod_secp256k1_add(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> BigIntVar<C> {
        self.mod_secp256k1_operation(left, right, DslIr::AddM)
    }

    pub fn mod_secp256k1_sub(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> BigIntVar<C> {
        self.mod_secp256k1_operation(left, right, DslIr::SubM)
    }

    pub fn mod_secp256k1_mul(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> BigIntVar<C> {
        self.mod_secp256k1_operation(left, right, DslIr::MulM)
    }

    pub fn mod_secp256k1_div(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> BigIntVar<C> {
        self.mod_secp256k1_operation(left, right, DslIr::DivM)
    }

    pub fn assert_mod_secp256k1_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) {
        self.assert_var_array_eq(left, right);
    }

    pub fn mod_secp256k1_is_zero(&mut self, bigint: &BigIntVar<C>) -> Var<C::N> {
        let result = self.eval(C::N::one());
        for i in 0..NUM_ELEMS {
            let elem = self.get(bigint, i);
            self.if_ne(elem, C::N::zero()).then(|builder| {
                builder.assign(&result, C::N::zero());
            });
        }

        result
    }

    pub fn mod_secp256k1_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> Var<C::N> {
        let diff = self.mod_secp256k1_sub(left, right);
        self.mod_secp256k1_is_zero(&diff)
    }

    pub fn if_mod_secp256k1_eq(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> IfBuilder<C> {
        let eq = self.mod_secp256k1_eq(left, right);
        self.if_eq(eq, C::N::one())
    }
}
