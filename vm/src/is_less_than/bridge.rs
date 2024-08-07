use afs_primitives::is_less_than::columns::IsLessThanIoCols;
use afs_stark_backend::interaction::InteractionBuilder;
use p3_field::AbstractField;

use super::IsLessThanVmAir;

impl IsLessThanVmAir {
    pub fn eval_interactions<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<AB::Var>,
    ) {
        builder.push_receive(self.bus_index, [io.x, io.y, io.less_than], AB::F::one());
    }
}
