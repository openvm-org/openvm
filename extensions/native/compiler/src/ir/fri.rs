use crate::ir::{Array, Builder, Config, Ext, Felt, Var};

impl<C: Config> Builder<C> {
    pub fn fri_single_reduced_opening_eval(
        &mut self,
        alpha: Ext<C::F, C::EF>,
        hint_id: Var<C::N>,
        is_init: Var<C::N>,
        at_x_array: &Array<C, Felt<C::F>>,
        at_z_array: &Array<C, Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF> {
        let result = self.uninit();
        self.operations.push(crate::ir::DslIr::FriReducedOpening(
            alpha,
            hint_id,
            is_init,
            at_x_array.clone(),
            at_z_array.clone(),
            result,
        ));
        result
    }
}
