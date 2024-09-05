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
pub const REPR_BITS: usize = 30;
/// Number of field elements used to represent a bigint.
pub const NUM_ELEMS: usize = 9;

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn eval_bigint(&mut self, bigint: BigUint) -> BigIntVar<C> {
        let array = self.dyn_array(NUM_ELEMS);

        let elems: Vec<C::N> = bigint_to_elems(bigint, REPR_BITS, NUM_ELEMS);
        for (i, &elem) in elems.iter().enumerate() {
            self.set(&array, i, elem);
        }

        array
    }

    pub fn uninit_bigint(&mut self) -> BigIntVar<C> {
        self.dyn_array(NUM_ELEMS)
    }

    fn mod_operation(
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

    pub fn secp256k1_coord_add(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::AddSecp256k1Coord)
    }

    pub fn secp256k1_coord_sub(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::SubSecp256k1Coord)
    }

    pub fn secp256k1_coord_mul(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::MulSecp256k1Coord)
    }

    pub fn secp256k1_coord_div(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::DivSecp256k1Coord)
    }

    pub fn assert_secp256k1_coord_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) {
        self.assert_var_array_eq(left, right);
    }

    pub fn secp256k1_coord_is_zero(&mut self, bigint: &BigIntVar<C>) -> Var<C::N> {
        let result = self.eval(C::N::one());
        for i in 0..NUM_ELEMS {
            let elem = self.get(bigint, i);
            self.if_ne(elem, C::N::zero()).then(|builder| {
                // FIXME: early break might improve performance.
                builder.assign(&result, C::N::zero());
            });
        }

        result
    }

    pub fn secp256k1_coord_set_to_zero(&mut self, bigint: &BigIntVar<C>) {
        for i in 0..NUM_ELEMS {
            self.set(bigint, i, C::N::zero());
        }
    }

    pub fn secp256k1_coord_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> Var<C::N> {
        let diff = self.secp256k1_coord_sub(left, right);
        self.secp256k1_coord_is_zero(&diff)
    }

    pub fn if_secp256k1_coord_eq(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> IfBuilder<C> {
        let eq = self.secp256k1_coord_eq(left, right);
        self.if_eq(eq, C::N::one())
    }

    pub fn secp256k1_scalar_add(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::AddSecp256k1Scalar)
    }

    pub fn secp256k1_scalar_sub(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::SubSecp256k1Scalar)
    }

    pub fn secp256k1_scalar_mul(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::MulSecp256k1Scalar)
    }

    pub fn secp256k1_scalar_div(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> BigIntVar<C> {
        self.mod_operation(left, right, DslIr::DivSecp256k1Scalar)
    }

    pub fn assert_secp256k1_scalar_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) {
        self.assert_var_array_eq(left, right);
    }

    pub fn secp256k1_scalar_is_zero(&mut self, bigint: &BigIntVar<C>) -> Var<C::N> {
        let result = self.eval(C::N::one());
        for i in 0..NUM_ELEMS {
            let elem = self.get(bigint, i);
            self.if_ne(elem, C::N::zero()).then(|builder| {
                builder.assign(&result, C::N::zero());
            });
        }

        result
    }

    pub fn secp256k1_scalar_eq(&mut self, left: &BigIntVar<C>, right: &BigIntVar<C>) -> Var<C::N> {
        let diff = self.secp256k1_scalar_sub(left, right);
        self.secp256k1_scalar_is_zero(&diff)
    }

    pub fn if_secp256k1_scalar_eq(
        &mut self,
        left: &BigIntVar<C>,
        right: &BigIntVar<C>,
    ) -> IfBuilder<C> {
        let eq = self.secp256k1_scalar_eq(left, right);
        self.if_eq(eq, C::N::one())
    }
}
