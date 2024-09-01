use stark_vm::hashes::keccak::hasher::KECCAK_DIGEST_U16S;

use super::{Array, Builder, Config, DslIr, Var};

impl<C: Config> Builder<C> {
    /// Computes the keccak256 hash of the given array.
    ///
    /// Reference: Section 5 of <https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf>
    ///
    /// Currently `array` will be auto-range checked to all be bytes.
    /// The output array is a length 16 array of u16 limbs,
    /// where each `u16` is converted from two bytes as **little-endian**.
    pub fn keccak256(&mut self, array: &Array<C, Var<C::N>>) -> Array<C, Var<C::N>> {
        let output = self.array::<Var<C::N>>(KECCAK_DIGEST_U16S);
        self.operations
            .push(DslIr::Keccak256(output.clone(), array.clone()));
        output
    }
}
