use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use crate::{arch::bus::ExecutionBus, memory::offline_checker::MemoryBridge};

// TODO: implement AIR

#[allow(dead_code)] // tmp
#[derive(Clone, Debug)]
pub struct UiAir {
    pub(super) execution_bus: ExecutionBus,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for UiAir
{
    fn width(&self) -> usize {
        0
    }
}

impl<AB: InteractionBuilder + AirBuilder> Air<AB> for UiAir {
    fn eval(&self, _builder: &mut AB) {}
}
