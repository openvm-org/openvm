use crate::ir::{Array, Builder, Config, Ext};

impl<C: Config> Builder<C> {
    pub fn fri_fold(
        &mut self,
        alpha: Ext<C::F, C::EF>,
        curr_alpha_pow: Ext<C::F, C::EF>,
        at_x_array: Array<C, Ext<C::F, C::EF>>,
        at_z_array: Array<C, Ext<C::F, C::EF>>,
    ) -> Ext<C::F, C::EF> {
        let result = self.uninit();
        self.operations.push(crate::ir::DslIr::FriFold(
            alpha,
            curr_alpha_pow,
            at_x_array,
            at_z_array,
            result,
        ));
        result
    }
}
