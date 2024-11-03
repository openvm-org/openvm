use crate::ir::{Array, Builder, Config, Ext, Felt};

impl<C: Config> Builder<C> {
    pub fn fri_mat_reduced_opening(
        &mut self,
        alpha: Ext<C::F, C::EF>,
        curr_alpha_pow: Ext<C::F, C::EF>,
        at_x_array: &Array<C, Felt<C::F>>,
        at_z_array: &Array<C, Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF> {
        let result = self.uninit();
        self.operations.push(crate::ir::DslIr::FriMatOpening(
            alpha,
            curr_alpha_pow,
            at_x_array.clone(),
            at_z_array.clone(),
            result,
        ));
        result
    }
}
