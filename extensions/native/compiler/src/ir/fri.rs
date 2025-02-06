use super::Var;
use crate::ir::{Array, Builder, Config, Ext, Felt};

impl<C: Config> Builder<C> {
    /// - `hint_id` is unconstrained index for the hint space.
    ///   What is read from the hint space depends on how many times the same `query_id` has been used before.
    ///
    /// The behavior is **different** depending on `ood_point_idx`:
    /// - if `ood_point_idx` is `0`, then `at_x_array` may be uninit and the appropriate hints from hint space will be written to `at_x_array`. The array will be of length equal to `at_z_array`.
    /// - otherwise, `at_x_array` must be initialized and will be read from.
    pub fn fri_single_reduced_opening_eval(
        &mut self,
        alpha: Ext<C::F, C::EF>,
        hint_id: Var<C>,
        ood_point_idx: Var<C>,
        at_x_array: &Array<C, Felt<C::F>>,
        at_z_array: &Array<C, Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF> {
        let result = self.uninit();
        self.operations.push(crate::ir::DslIr::FriReducedOpening(
            alpha,
            hint_id,
            ood_point_idx,
            at_x_array.clone(),
            at_z_array.clone(),
            result,
        ));
        result
    }
}
