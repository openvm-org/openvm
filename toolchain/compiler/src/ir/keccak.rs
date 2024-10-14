use stark_vm::intrinsics::hashes::keccak::hasher::KECCAK_DIGEST_BYTES;

use super::{Array, Builder, Config, DslIr, Var};

impl<C: Config> Builder<C> {
    /// Computes the keccak256 hash of the given array.
    ///
    /// Reference: Section 5 of <https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.202.pdf>
    ///
    /// Currently input `array` will be auto-range checked to all be bytes.
    /// The output array is a length 32 array of u8 bytes.
    ///
    /// **SAFETY:** The output array is **not** range checked to be bytes.
    pub fn keccak256(&mut self, array: &Array<C, Var<C::N>>) -> Array<C, Var<C::N>> {
        let output = self.array::<Var<C::N>>(KECCAK_DIGEST_BYTES);
        self.operations
            .push(DslIr::Keccak256(output.clone(), array.clone()));
        output
    }
}
