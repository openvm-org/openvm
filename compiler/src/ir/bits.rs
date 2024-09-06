use p3_field::AbstractField;

use super::{Array, BigUintVar, Builder, Config, DslIr, Felt, MemIndex, RVar, Var};

pub const NUM_BITS: usize = 31;

impl<C: Config> Builder<C> {
    /// Converts a variable to bits.
    pub fn num2bits_v(&mut self, num: Var<C::N>, num_bits: u32) -> Array<C, Var<C::N>> {
        self.push(DslIr::HintBitsV(num, num_bits));

        let output = self.dyn_array::<Var<_>>(num_bits as usize);

        let sum: Var<_> = self.eval(C::N::zero());
        for i in 0..num_bits as usize {
            let index = MemIndex {
                index: i.into(),
                offset: 0,
                size: 1,
            };
            self.push(DslIr::StoreHintWord(output.ptr(), index));

            let bit = self.get(&output, i);
            self.assert_var_eq(bit * (bit - C::N::one()), C::N::zero());
            self.assign(&sum, sum + bit * C::N::from_canonical_u32(1 << i));
        }

        // TODO: There is an edge case where the witnessed bits may slightly overflow and cause
        // the output to be incorrect. This is a known issue and will be fixed in the future.
        self.assert_var_eq(sum, num);

        output
    }

    pub fn num2bits_biguint(&mut self, biguint: &BigUintVar<C>) -> Array<C, Var<C::N>> {
        let repr_size = self.bigint_repr_size;
        let num_limbs = (256 + repr_size - 1) / repr_size;
        let bits = self.dyn_array((num_limbs * repr_size) as usize);
        for i in 0..num_limbs as usize {
            let limb = self.get(biguint, i);
            let limb_bits = self.num2bits_v(limb, repr_size);
            for j in 0..repr_size as usize {
                let val = self.get(&limb_bits, j);
                self.set_value(&bits, j + i * repr_size as usize, val);
            }
        }
        bits
    }

    /// Converts a variable to bits inside a circuit.
    pub fn num2bits_v_circuit(&mut self, num: Var<C::N>, bits: usize) -> Vec<Var<C::N>> {
        let mut output = Vec::new();
        for _ in 0..bits {
            output.push(self.uninit());
        }

        self.push(DslIr::CircuitNum2BitsV(num, bits, output.clone()));

        output
    }

    /// Converts a felt to bits.
    pub fn num2bits_f(&mut self, num: Felt<C::F>, num_bits: u32) -> Array<C, Var<C::N>> {
        self.push(DslIr::HintBitsF(num, num_bits));

        let output = self.dyn_array::<Var<_>>(num_bits as usize);

        let sum: Felt<_> = self.eval(C::F::zero());
        for i in 0..num_bits as usize {
            let index = MemIndex {
                index: i.into(),
                offset: 0,
                size: 1,
            };
            self.push(DslIr::StoreHintWord(output.ptr(), index));

            let bit = self.get(&output, i);
            self.assert_var_eq(bit * (bit - C::N::one()), C::N::zero());
            self.if_eq(bit, C::N::one()).then(|builder| {
                builder.assign(&sum, sum + C::F::from_canonical_u32(1 << i));
            });
        }

        // TODO: There is an edge case where the witnessed bits may slightly overflow and cause
        // the output to be incorrect. This is a known issue and will be fixed in the future.
        self.assert_felt_eq(sum, num);

        output
    }

    /// Converts a felt to bits inside a circuit.
    pub fn num2bits_f_circuit(&mut self, num: Felt<C::F>) -> Vec<Var<C::N>> {
        let mut output = Vec::new();
        for _ in 0..32 {
            output.push(self.uninit());
        }

        self.push(DslIr::CircuitNum2BitsF(num, output.clone()));

        output
    }

    /// Convert bits to a variable.
    pub fn bits2num_v(&mut self, bits: &Array<C, Var<C::N>>) -> Var<C::N> {
        let num: Var<_> = self.eval(C::N::zero());
        let power: Var<_> = self.eval(C::N::one());
        self.range(0, bits.len()).for_each(|i, builder| {
            let bit = builder.get(bits, i);
            builder.assign(&num, num + bit * power);
            builder.assign(&power, power * C::N::from_canonical_u32(2));
        });
        num
    }

    /// Convert bits to a variable inside a circuit.
    pub fn bits2num_v_circuit(&mut self, bits: &[Var<C::N>]) -> Var<C::N> {
        let result: Var<_> = self.eval(C::N::zero());
        for i in 0..bits.len() {
            self.assign(&result, result + bits[i] * C::N::from_canonical_u32(1 << i));
        }
        result
    }

    /// Convert bits to a felt.
    pub fn bits2num_f(&mut self, bits: &Array<C, Var<C::N>>, num_bits: u32) -> Felt<C::F> {
        let num: Felt<_> = self.eval(C::F::zero());
        for i in 0..num_bits as usize {
            let bit = self.get(bits, i);
            // Add `bit * 2^i` to the sum.
            self.if_eq(bit, C::N::one()).then(|builder| {
                builder.assign(&num, num + C::F::from_canonical_u32(1 << i));
            });
        }
        num
    }

    /// Reverse a list of bits.
    ///
    /// SAFETY: calling this function with `bit_len` greater [`NUM_BITS`] will result in undefined
    /// behavior.
    ///
    /// Reference: [p3_util::reverse_bits_len]
    pub fn reverse_bits_len(
        &mut self,
        index_bits: &Array<C, Var<C::N>>,
        bit_len: impl Into<RVar<C::N>>,
    ) -> Array<C, Var<C::N>> {
        let bit_len = bit_len.into();
        let num_bits = NUM_BITS;

        let result_bits = self.dyn_array::<Var<_>>(num_bits);
        self.range(0, bit_len).for_each(|i, builder| {
            let idx = builder.eval_expr(bit_len - i - RVar::one());
            let entry = builder.get(index_bits, idx);
            builder.set_value(&result_bits, i, entry);
        });

        let zero = self.eval(C::N::zero());
        self.range(bit_len, num_bits).for_each(|i, builder| {
            builder.set_value(&result_bits, i, zero);
        });

        result_bits
    }

    /// Reverse a list of bits inside a circuit.
    ///
    /// SAFETY: calling this function with `bit_len` greater [`NUM_BITS`] will result in undefined
    /// behavior.
    ///
    /// Reference: [p3_util::reverse_bits_len]
    pub fn reverse_bits_len_circuit(
        &mut self,
        index_bits: Vec<Var<C::N>>,
        bit_len: usize,
    ) -> Vec<Var<C::N>> {
        assert!(bit_len <= NUM_BITS);
        let mut result_bits = Vec::new();
        for i in 0..bit_len {
            let idx = bit_len - i - 1;
            result_bits.push(index_bits[idx]);
        }
        result_bits
    }
}
