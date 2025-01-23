use crate::ir::{Array, Builder, Config, Ext, Felt};

impl<C: Config> Builder<C> {
    pub fn fri_single_reduced_opening_eval(
        &mut self,
        alpha: Ext<C::F, C::EF>,
        at_x_array: &Array<C, Felt<C::F>>,
        at_z_array: &Array<C, Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF> {
        let result = self.uninit();
        self.operations.push(crate::ir::DslIr::FriReducedOpening(
            alpha,
            at_x_array.clone(),
            at_z_array.clone(),
            result,
        ));
        result
    }
}
