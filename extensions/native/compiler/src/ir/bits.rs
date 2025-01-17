use openvm_stark_backend::p3_field::FieldAlgebra;

use super::{Array, Builder, Config, DslIr, Felt, MemIndex, Var};

impl<C: Config> Builder<C> {
    /// Converts a variable to bits inside a circuit.
    pub fn num2bits_v_circuit(&mut self, num: Var<C::N>, bits: usize) -> Vec<Var<C::N>> {
        let mut output = Vec::new();
        for _ in 0..bits {
            output.push(self.uninit());
        }

        self.push(DslIr::CircuitNum2BitsV(num, bits, output.clone()));

        output
    }

    /// Converts a felt to bits. Only works for C::F = BabyBear
    pub fn num2bits_f(&mut self, num: Felt<C::F>, num_bits: u32) -> Array<C, Var<C::N>> {
        assert!(TypeId::of::<C::F>() == BabyBear::F::TYPE_ID);

        self.push(DslIr::HintBitsF(num, num_bits));
        let output = self.dyn_array::<Felt<_>>(num_bits as usize);

        let sum: Felt<_> = self.eval(C::F::ZERO);
        // will be used to compute b_0 + ... + b_16 * 2^16
        let prefix_sum: Felt<_> = self.eval(C::F::ZERO);
        // will be used to compute b_17 + ... + b_30
        let suffix_bit_sum: Felt<_> = self.eval(C::F::ZERO);
        for i in 0..num_bits as usize {
            let index = MemIndex {
                index: i.into(),
                offset: 0,
                size: 1,
            };
            self.push(DslIr::StoreHintWord(output.ptr(), index));

            let bit = self.get(&output, i);
            self.assert_felt_eq(bit * (bit - C::F::ONE), C::F::ZERO);
            self.assign(&sum, sum + bit * C::F::from_canonical_u32(1 << i));
            if i == 16 {
                self.assign(&prefix_sum, sum);
            }
            if i > 16 {
                self.assign(&suffix_bit_sum, suffix_bit_sum + bit);
            }
        }
        self.assert_felt_eq(sum, num);

        // Check that the bits represent the number without overflow.
        // If F is BabyBear, then any element of F can be represented either as:
        //    * 2^30 + ... + 2^x + y for y in [0, 2^(x - 1)) and x > 17
        //    * 2^30 + ... + 2^17
        // To check that bits b[0], ..., b[30] represent b[0] + ... + b[30] * 2^30 without overflow,
        // we may check that:
        //    * if b_17 + ... + b_30 = 14, then b_0 + ... + b_16 * 2^16 = 0
        let suffix_bit_sum_var = self.cast_felt_to_var(suffix_bit_sum);
        self.if_eq(suffix_bit_sum_var, C::N::from_canonical_u32(14))
            .then(|builder| {
                builder.assert_felt_eq(prefix_sum, C::F::ZERO);
            });

        // Cast Array<C, Felt<C::F>> to Array<C, Var<C::N>>
        Array::Dyn(output.ptr(), output.len())
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

    /// Convert bits to a variable inside a circuit.
    pub fn bits2num_v_circuit(&mut self, bits: &[Var<C::N>]) -> Var<C::N> {
        let result: Var<_> = self.eval(C::N::ZERO);
        for i in 0..bits.len() {
            self.assign(&result, result + bits[i] * C::N::from_canonical_u32(1 << i));
        }
        result
    }
}
