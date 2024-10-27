use afs_primitives::bigint::utils::big_uint_to_num_limbs;
use num_bigint_dig::BigUint;
use num_traits::Zero;
use p3_field::{AbstractField, PrimeField64};
use stark_vm::{
    arch::Modulus,
    intrinsics::modular::{SECP256K1_COORD_PRIME, SECP256K1_SCALAR_PRIME},
};

use super::{
    utils::{LIMB_BITS, NUM_LIMBS},
    Array, Builder, Config, DslIr, IfBuilder, Var,
};

pub type BigUintVar<C> = Array<C, Var<<C as Config>::N>>;

impl<C: Config> BigUintVar<C> {
    pub fn ptr_fp(&self) -> i32 {
        match self {
            Array::Fixed(_) => panic!(),
            Array::Dyn(ptr, _) => ptr.fp(),
        }
    }
}

impl<C: Config> Builder<C>
where
    C::N: PrimeField64,
{
    pub fn eval_biguint(&mut self, biguint: BigUint) -> BigUintVar<C> {
        let array = self.dyn_array(NUM_LIMBS);

        let elems: Vec<C::N> = big_uint_to_num_limbs(&biguint, LIMB_BITS, NUM_LIMBS)
            .into_iter()
            .map(C::N::from_canonical_usize)
            .collect();
        for (i, &elem) in elems.iter().enumerate() {
            self.set(&array, i, elem);
        }

        array
    }

    pub fn uninit_biguint(&mut self) -> BigUintVar<C> {
        self.dyn_array(NUM_LIMBS)
    }

    fn mod_operation(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
        modulus: BigUint,
        operation: impl Fn(BigUint, BigUintVar<C>, BigUintVar<C>, BigUintVar<C>) -> DslIr<C>,
    ) -> BigUintVar<C> {
        let dst = self.dyn_array(NUM_LIMBS);
        self.operations
            .push(operation(modulus, dst.clone(), left.clone(), right.clone()));
        dst
    }

    pub fn secp256k1_coord_add(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Coord.prime(),
            DslIr::ModularAdd,
        )
    }

    pub fn secp256k1_coord_sub(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Coord.prime(),
            DslIr::ModularSub,
        )
    }

    pub fn secp256k1_coord_mul(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Coord.prime(),
            DslIr::ModularMul,
        )
    }

    pub fn secp256k1_coord_div(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Coord.prime(),
            DslIr::ModularDiv,
        )
    }

    pub fn assert_secp256k1_coord_eq(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) {
        let res = self.secp256k1_coord_eq(left, right);
        self.assert_var_eq(res, C::N::one());
    }

    pub fn secp256k1_coord_is_zero(&mut self, biguint: &BigUintVar<C>) -> Var<C::N> {
        // TODO: either EqU256 needs to support address space 0 or we just need better pointer handling here.
        let ret_arr = self.array(1);
        // FIXME: reuse constant zero.
        let big_zero = self.eval_biguint(BigUint::zero());
        self.operations
            .push(DslIr::EqualTo256(ret_arr.ptr(), biguint.clone(), big_zero));
        let ret: Var<_> = self.get(&ret_arr, 0);
        self.if_ne(ret, C::N::one()).then(|builder| {
            // FIXME: reuse constant.
            let big_n = builder.eval_biguint(SECP256K1_COORD_PRIME.clone());
            builder
                .operations
                .push(DslIr::EqualTo256(ret_arr.ptr(), biguint.clone(), big_n));
            let _ret: Var<_> = builder.get(&ret_arr, 0);
            builder.assign(&ret, _ret);
        });
        ret
    }

    pub fn secp256k1_coord_set_to_zero(&mut self, biguint: &BigUintVar<C>) {
        for i in 0..NUM_LIMBS {
            self.set(biguint, i, C::N::zero());
        }
    }

    pub fn secp256k1_coord_eq(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) -> Var<C::N> {
        let diff = self.secp256k1_coord_sub(left, right);
        self.secp256k1_coord_is_zero(&diff)
    }

    pub fn if_secp256k1_coord_eq(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> IfBuilder<C> {
        let eq = self.secp256k1_coord_eq(left, right);
        self.if_eq(eq, C::N::one())
    }

    pub fn secp256k1_scalar_add(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Scalar.prime(),
            DslIr::ModularAdd,
        )
    }

    pub fn secp256k1_scalar_sub(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Scalar.prime(),
            DslIr::ModularSub,
        )
    }

    pub fn secp256k1_scalar_mul(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Scalar.prime(),
            DslIr::ModularMul,
        )
    }

    pub fn secp256k1_scalar_div(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> BigUintVar<C> {
        self.mod_operation(
            left,
            right,
            Modulus::Secp256k1Scalar.prime(),
            DslIr::ModularDiv,
        )
    }

    pub fn assert_secp256k1_scalar_eq(&mut self, left: &BigUintVar<C>, right: &BigUintVar<C>) {
        let res = self.secp256k1_scalar_eq(left, right);
        self.assert_var_eq(res, C::N::one());
    }

    pub fn secp256k1_scalar_is_zero(&mut self, biguint: &BigUintVar<C>) -> Var<C::N> {
        // TODO: either EqU256 needs to support address space 0 or we just need better pointer handling here.
        let ret_arr = self.array(1);
        // FIXME: reuse constant zero.
        let big_zero = self.eval_biguint(BigUint::zero());
        self.operations
            .push(DslIr::EqualTo256(ret_arr.ptr(), biguint.clone(), big_zero));
        let ret: Var<_> = self.get(&ret_arr, 0);
        self.if_ne(ret, C::N::one()).then(|builder| {
            // FIXME: reuse constant.
            let big_n = builder.eval_biguint(SECP256K1_SCALAR_PRIME.clone());
            builder
                .operations
                .push(DslIr::EqualTo256(ret_arr.ptr(), biguint.clone(), big_n));
            let _ret: Var<_> = builder.get(&ret_arr, 0);
            builder.assign(&ret, _ret);
        });
        ret
    }

    pub fn secp256k1_scalar_eq(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> Var<C::N> {
        let diff = self.secp256k1_scalar_sub(left, right);
        self.secp256k1_scalar_is_zero(&diff)
    }

    pub fn if_secp256k1_scalar_eq(
        &mut self,
        left: &BigUintVar<C>,
        right: &BigUintVar<C>,
    ) -> IfBuilder<C> {
        let eq = self.secp256k1_scalar_eq(left, right);
        self.if_eq(eq, C::N::one())
    }
}
