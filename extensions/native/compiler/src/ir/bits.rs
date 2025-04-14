use std::{any::TypeId, array};

use openvm_stark_backend::p3_field::FieldAlgebra;
use openvm_stark_sdk::p3_baby_bear::BabyBear;

use super::{Array, Builder, Config, DslIr, Felt, MemIndex, Var};

impl<C: Config> Builder<C> {
    /// Converts a felt to bits. Will result in a failed assertion if `num` has more than `num_bits`
    /// bits. Only works for C::F = BabyBear
    pub fn num2bits_f(&mut self, num: Felt<C::F>, num_bits: u32) -> Array<C, Var<C::N>> {
        assert_eq!(TypeId::of::<C::F>(), TypeId::of::<BabyBear>());

        self.push(DslIr::HintBitsF(num, num_bits));
        let output = self.dyn_array::<Felt<_>>(num_bits as usize);

        let sum: Felt<_> = self.eval(C::F::ZERO);
        // if `num_bits >= 27`, this will be used to compute b_0 + ... + b_26 * 2^26
        // otherwise, this will be 0
        let prefix_sum: Felt<_> = self.eval(C::F::ZERO);
        // if `num_bits >= 27`, this will be used to compute b_27 + ... + b_30
        // otherwise, this will be 0
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
            if i == 26 {
                self.assign(&prefix_sum, sum);
            }
            if i > 26 {
                self.assign(&suffix_bit_sum, suffix_bit_sum + bit);
            }
        }
        self.assert_felt_eq(sum, num);

        // Check that the bits represent the number without overflow.
        // If F is BabyBear, then any element of F can be represented either as:
        //    * 2^30 + ... + 2^x + y for y in [0, 2^(x - 1)) and 27 < x <= 30
        //    * 2^30 + ... + 2^27
        //    * y for y in [0, 2^27)
        // To check that bits `b[0], ..., b[30]` represent `num = b[0] + ... + b[30] * 2^30` without
        // overflow, we may check that:
        //    * if `num_bits < 27`, then `b[30] = 0`, so overflow is impossible. In this case,
        //      `suffix_bit_sum = 0`, so the check below passes.
        //    * if `num_bits >= 27`, then we must check: if `suffix_bit_sum = b[27] + ... + b[30] =
        //      4`, then `prefix_sum = b[0] + ... + b[26] * 2^26 = 0`
        let suffix_bit_sum_var = self.cast_felt_to_var(suffix_bit_sum);
        self.if_eq(suffix_bit_sum_var, C::N::from_canonical_u32(4))
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

    /// Decompose a Var into 64-bit Felt limbs.
    pub fn var_to_64bits_f_circuit(&mut self, value: Var<C::N>) -> [Felt<C::F>; 4] {
        let ret = array::from_fn(|_| self.uninit());
        self.push(DslIr::CircuitVarTo64BitsF(value, ret));
        ret
    }
}
