use std::ops::{Add, Mul, MulAssign};

use p3_field::{AbstractExtensionField, AbstractField, PrimeField};

use super::{
    Array, Builder, CanSelect, Config, DslIr, Ext, Felt, MemIndex, RVar, SymbolicExt, Var, Variable,
};

pub const NUM_LIMBS: usize = 32;
pub const LIMB_BITS: usize = 8;

pub type BigUintVar<C> = Array<C, Var<<C as Config>::N>>;

/// Converts a prime field element to a usize.
pub fn prime_field_to_usize<F: PrimeField>(x: F) -> usize {
    let bu = x.as_canonical_biguint();
    let digits = bu.to_u64_digits();
    if digits.is_empty() {
        return 0;
    }
    assert!(digits.len() == 1, "Prime field element is too large");
    digits[0] as usize
}

impl<C: Config> Builder<C> {
    /// The generator for the field.
    ///
    /// Reference: [p3_baby_bear::BabyBear]
    pub fn generator(&mut self) -> Felt<C::F> {
        self.eval(C::F::from_canonical_u32(31))
    }

    /// Select a variable based on a condition.
    pub fn select_v(&mut self, cond: Var<C::N>, a: Var<C::N>, b: Var<C::N>) -> Var<C::N> {
        let c = self.uninit();
        if self.flags.static_only {
            self.operations.push(DslIr::CircuitSelectV(cond, a, b, c));
        } else {
            self.if_eq(cond, C::N::one()).then_or_else(
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
            self.if_eq(cond, C::N::one()).then_or_else(
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
            self.if_eq(cond, C::N::one()).then_or_else(
                |builder| builder.assign(&c, a),
                |builder| builder.assign(&c, b),
            );
        }
        c
    }

    /// Exponentiates a variable to a power of two.
    pub fn exp_power_of_2<V: Variable<C>, E: Into<V::Expression>>(
        &mut self,
        e: E,
        power_log: usize,
    ) -> V
    where
        V::Expression: MulAssign<V::Expression> + Clone,
    {
        let mut e = e.into();
        for _ in 0..power_log {
            e *= e.clone();
        }
        self.eval(e)
    }

    /// Exponentiates a variable to an array of bits in little endian.
    pub fn exp_bits<V>(&mut self, x: V, power_bits: &Array<C, Var<C::N>>) -> V
    where
        V::Expression: AbstractField,
        V: Copy + Mul<Output = V::Expression> + Variable<C>,
    {
        let result: V = self.eval(V::Expression::one());
        let power_f: V = self.eval(x);
        self.range(0, power_bits.len()).for_each(|i, builder| {
            let bit = builder.get(power_bits, i);
            builder
                .if_eq(bit, C::N::one())
                .then(|builder| builder.assign(&result, result * power_f));
            builder.assign(&power_f, power_f * power_f);
        });
        result
    }

    /// Exponentiates a felt to a list of bits in little endian.
    pub fn exp_f_bits(&mut self, x: Felt<C::F>, power_bits: Vec<Var<C::N>>) -> Felt<C::F> {
        let mut result = self.eval(C::F::one());
        let mut power_f: Felt<_> = self.eval(x);
        for i in 0..power_bits.len() {
            let bit = power_bits[i];
            let tmp = self.eval(result * power_f);
            result = self.select_f(bit, tmp, result);
            power_f = self.eval(power_f * power_f);
        }
        result
    }

    /// Exponentiates a extension to a list of bits in little endian.
    pub fn exp_e_bits(
        &mut self,
        x: Ext<C::F, C::EF>,
        power_bits: Vec<Var<C::N>>,
    ) -> Ext<C::F, C::EF> {
        let mut result = self.eval(SymbolicExt::from_f(C::EF::one()));
        let mut power_f: Ext<_, _> = self.eval(x);
        for i in 0..power_bits.len() {
            let bit = power_bits[i];
            let tmp = self.eval(result * power_f);
            result = self.select_ef(bit, tmp, result);
            power_f = self.eval(power_f * power_f);
        }
        result
    }

    /// Exponentiates a variable to a list of reversed bits with a given length.
    ///
    /// Reference: [p3_util::reverse_bits_len]
    pub fn exp_reverse_bits_len<V>(
        &mut self,
        x: V,
        power_bits: &Array<C, Var<C::N>>,
        bit_len: impl Into<RVar<C::N>>,
    ) -> V
    where
        V::Expression: AbstractField,
        V: Copy + Mul<Output = V::Expression> + Variable<C> + CanSelect<C>,
    {
        let result: V = self.eval(V::Expression::one());
        let power_f: V = self.eval(x);
        let bit_len = bit_len.into();
        let bit_len_plus_one = self.eval_expr(bit_len + C::N::one());
        let one_var: V = self.eval(V::Expression::one());

        self.range(RVar::one(), bit_len_plus_one)
            .for_each(|i, builder| {
                let index = builder.eval_expr(bit_len - i);
                let bit = builder.get(power_bits, index);
                let mul = V::select(builder, bit, power_f, one_var);
                builder.assign(&result, result * mul);
                builder.assign(&power_f, power_f * power_f);
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

    /// Exponentiates a variable to a list of bits in little endian inside a circuit.
    pub fn exp_power_of_2_v_circuit<V>(
        &mut self,
        base: impl Into<V::Expression>,
        power_log: usize,
    ) -> V
    where
        V: Copy + Mul<Output = V::Expression> + Variable<C>,
    {
        let mut result: V = self.eval(base);
        for _ in 0..power_log {
            result = self.eval(result * result)
        }
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
        assert!(arr.len() <= <C::EF as AbstractExtensionField<C::F>>::D);
        let mut res = SymbolicExt::from_f(C::EF::zero());
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

    pub fn uninit_biguint(&mut self) -> BigUintVar<C> {
        self.dyn_array(NUM_LIMBS)
    }
}
