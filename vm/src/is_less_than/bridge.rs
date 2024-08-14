use afs_stark_backend::interaction::InteractionBuilder;

use super::{columns::IsLessThanVmCols, IsLessThanVmAir};

impl IsLessThanVmAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        cols: IsLessThanVmCols<AB::Var>,
    ) {
        builder.push_receive(
            self.bus_index,
            [
                cols.internal.io.x,
                cols.internal.io.y,
                cols.internal.io.less_than,
            ],
            cols.is_enabled,
        );
    }
}
