use std::ops::{Add, Mul};

use openvm_native_compiler_derive::iter_zip;
use openvm_stark_backend::p3_field::{FieldAlgebra, FieldExtensionAlgebra, PrimeField};

use super::{
    Array, ArrayLike, Builder, CanSelect, Config, DslIr, Ext, Felt, MemIndex, RVar, SymbolicExt,
    Var, Variable,
};

pub const NUM_LIMBS: usize = 32;
pub const LIMB_BITS: usize = 8;

/// Converts a prime field element to a usize.
pub fn prime_field_to_usize<F: PrimeField>(x: F) -> usize {
    let biguint = x.as_canonical_biguint();
    let digits = biguint.to_u64_digits();
    if digits.is_empty() {
        return 0;
    }
    assert!(digits.len() == 1, "Prime field element is too large");
    digits[0] as usize
}

impl<C: Config> Builder<C> {
    /// The generator for the field.
    ///
    /// Reference: [`openvm_stark_sdk::p3_baby_bear::BabyBear`]
    pub fn generator(&mut self) -> Felt<C::F> {
        self.eval(C::F::from_canonical_u32(31))
    }

    /// Select a variable based on a condition.
    pub fn select_v(&mut self, cond: Var<C::N>, a: Var<C::N>, b: Var<C::N>) -> Var<C::N> {
        let c = self.uninit();
        if self.flags.static_only {
            self.operations.push(DslIr::CircuitSelectV(cond, a, b, c));
        } else {
            self.if_eq(cond, C::N::ONE).then_or_else(
                |builder| builder.assign(&c, a),
                |builder| builder.assign(&c, b),
            );
        }
        c
    }

    /// Select a felt based on a condition.
    pub fn select_f(&mut self, cond: Var<C::N>, a: Felt<C::F>, b: Felt<C::F>) -> Felt<C::F> {
        let c = self.uninit();
        if self.flags.static_only {
            self.operations.push(DslIr::CircuitSelectF(cond, a, b, c));
        } else {
            self.if_eq(cond, C::N::ONE).then_or_else(
                |builder| builder.assign(&c, a),
                |builder| builder.assign(&c, b),
            );
        }
        c
    }

    /// Select an extension based on a condition.
    pub fn select_ef(
        &mut self,
        cond: Var<C::N>,
        a: Ext<C::F, C::EF>,
        b: Ext<C::F, C::EF>,
    ) -> Ext<C::F, C::EF> {
        let c = self.uninit();
        if self.flags.static_only {
            self.operations.push(DslIr::CircuitSelectE(cond, a, b, c));
        } else {
            self.if_eq(cond, C::N::ONE).then_or_else(
                |builder| builder.assign(&c, a),
                |builder| builder.assign(&c, b),
            );
        }
        c
    }

    /// Exponentiates a variable to a list of big endian bits with a given length.
    ///
    /// Example: if power_bits = [1, 0, 1, 0], then the result should be x^8 * x^2 = x^10.
    pub fn exp_bits_big_endian<V>(&mut self, x: V, power_bits: &Array<C, Var<C::N>>) -> V
    where
        V::Expression: FieldAlgebra,
        V: Copy + Mul<Output = V::Expression> + Variable<C> + CanSelect<C>,
    {
        let result: V = self.eval(V::Expression::ONE);
        let power_f: V = self.eval(x);
        let one_var: V = self.eval(V::Expression::ONE);

        // Implements a square-and-multiply algorithm.
        iter_zip!(self, power_bits).for_each(|ptr_vec, builder| {
            let bit = builder.iter_ptr_get(power_bits, ptr_vec[0]);
            builder.assign(&result, result * result);
            let mul = V::select(builder, bit, power_f, one_var);
            builder.assign(&result, result * mul);
        });

        result
    }

    /// Exponentiates a variable to a list of bits in little endian.
    pub fn exp_power_of_2_v<V>(
        &mut self,
        base: impl Into<V::Expression>,
        power_log: impl Into<RVar<C::N>>,
    ) -> V
    where
        V: Variable<C> + Copy + Mul<Output = V::Expression>,
    {
        let result: V = self.eval(base);
        let power_log = power_log.into();
        self.range(0, power_log)
            .for_each(|_, builder| builder.assign(&result, result * result));
        result
    }

    /// Multiplies `base` by `2^{log_power}`.
    pub fn sll<V>(&mut self, base: impl Into<V::Expression>, shift: RVar<C::N>) -> V
    where
        V: Variable<C> + Clone + Add<Output = V::Expression>,
    {
        let result: V = self.eval(base);
        self.range(0, shift)
            .for_each(|_, builder| builder.assign(&result, result.clone() + result.clone()));
        result
    }

    /// Creates an ext from a slice of felts.
    pub fn ext_from_base_slice(&mut self, arr: &[Felt<C::F>]) -> Ext<C::F, C::EF> {
        assert!(arr.len() <= <C::EF as FieldExtensionAlgebra<C::F>>::D);
        let mut res = SymbolicExt::from_f(C::EF::ZERO);
        for i in 0..arr.len() {
            res += arr[i] * SymbolicExt::from_f(C::EF::monomial(i));
        }
        self.eval(res)
    }

    pub fn felts2ext(&mut self, felts: &[Felt<C::F>]) -> Ext<C::F, C::EF> {
        assert_eq!(felts.len(), 4);
        let out: Ext<C::F, C::EF> = self.uninit();
        self.push(DslIr::CircuitFelts2Ext(felts.try_into().unwrap(), out));
        out
    }

    /// Converts an ext to a slice of felts.
    pub fn ext2felt(&mut self, value: Ext<C::F, C::EF>) -> Array<C, Felt<C::F>> {
        if self.flags.static_only {
            let felts = self.ext2felt_circuit(value);
            self.vec(felts.to_vec())
        } else {
            let result = self.array(C::EF::D);
            let index = MemIndex {
                index: RVar::zero(),
                offset: 0,
                size: C::EF::D,
            };
            if let Array::Dyn(ptr, _) = &result {
                self.store(*ptr, index, value);
            } else {
                unreachable!()
            }
            result
        }
    }

    /// Converts an ext to a slice of felts inside a circuit.
    pub fn ext2felt_circuit(&mut self, value: Ext<C::F, C::EF>) -> [Felt<C::F>; 4] {
        let a = self.uninit();
        let b = self.uninit();
        let c = self.uninit();
        let d = self.uninit();
        self.operations
            .push(DslIr::CircuitExt2Felt([a, b, c, d], value));
        [a, b, c, d]
    }
    pub fn felt_reduce_circuit(&mut self, value: Felt<C::F>) {
        self.operations.push(DslIr::CircuitFeltReduce(value));
    }
    pub fn ext_reduce_circuit(&mut self, value: Ext<C::F, C::EF>) {
        self.operations.push(DslIr::CircuitExtReduce(value));
    }
}
