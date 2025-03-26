use crate::ir::{Array, Builder, Config, DslIr, Ext, Felt, Usize, Var};

impl<C: Config> Builder<C> {
    /// - Requires `dimensions.len() == opened_values.len()`
    /// - `proof` is an array of arrays where inner arrays are of length `CHUNK`
    /// - `commit.len() = CHUNK`
    pub fn verify_batch_felt(
        &mut self,
        dimensions: &Array<C, Usize<C::F>>,
        opened_values: &Array<C, Array<C, Felt<C::F>>>,
        proof_id: Var<C::N>,
        index_bits: &Array<C, Var<C::N>>,
        commit: &Array<C, Felt<C::F>>,
    ) {
        self.push(DslIr::VerifyBatchFelt(
            dimensions.clone(),
            opened_values.clone(),
            proof_id,
            index_bits.clone(),
            commit.clone(),
        ));
    }

    /// Version of [`Self::verify_batch_felt`] where `opened_values` are extension field elements.
    /// - Requires `dimensions.len() == opened_values.len()`
    /// - `proof` is an array of arrays where inner arrays are of length `CHUNK`
    /// - `commit.len() = CHUNK`
    pub fn verify_batch_ext(
        &mut self,
        dimensions: &Array<C, Usize<C::F>>,
        opened_values: &Array<C, Array<C, Ext<C::F, C::EF>>>,
        proof_id: Var<C::N>,
        index_bits: &Array<C, Var<C::N>>,
        commit: &Array<C, Felt<C::F>>,
    ) {
        self.push(DslIr::VerifyBatchExt(
            dimensions.clone(),
            opened_values.clone(),
            proof_id,
            index_bits.clone(),
            commit.clone(),
        ));
    }
}
