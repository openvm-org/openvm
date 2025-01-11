use openvm_stark_backend::p3_field::FieldAlgebra;

use super::{Array, Builder, Config, DslIr, Felt, MemIndex, Var};

impl<C: Config> Builder<C> {
    /// Converts a variable to bits.
    pub fn num2bits_v(&mut self, num: Var<C::N>, num_bits: u32) -> Array<C, Var<C::N>> {
        self.push(DslIr::HintBitsV(num, num_bits));

        let output = self.dyn_array::<Var<_>>(num_bits as usize);

        let sum: Var<_> = self.eval(C::N::ZERO);
        for i in 0..num_bits as usize {
            let index = MemIndex {
                index: i.into(),
                offset: 0,
                size: 1,
            };
            self.push(DslIr::StoreHintWord(output.ptr(), index));

            let bit = self.get(&output, i);
            self.assert_var_eq(bit * (bit - C::N::ONE), C::N::ZERO);
            self.assign(&sum, sum + bit * C::N::from_canonical_u32(1 << i));
        }

        // FIXME: There is an edge case where the witnessed bits may slightly overflow and cause
        // the output to be incorrect.
        self.assert_var_eq(sum, num);

        output
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

        let output = self.dyn_array::<Felt<_>>(num_bits as usize);

        let sum: Felt<_> = self.eval(C::F::ZERO);
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
        }

        // FIXME: There is an edge case where the witnessed bits may slightly overflow and cause
        // the output to be incorrect.
        self.assert_felt_eq(sum, num);

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
