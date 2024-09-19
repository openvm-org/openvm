use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;

use crate::{arch::bus::ExecutionBus, memory::offline_checker::MemoryBridge};

// TODO: implement AIR

#[allow(dead_code)] // tmp
#[derive(Clone, Debug)]
pub struct ShiftAir<const NUM_LIMBS: usize, const LIMB_BITS: usize> {
    pub(super) execution_bus: ExecutionBus,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field, const NUM_LIMBS: usize, const LIMB_BITS: usize> BaseAir<F>
    for ShiftAir<NUM_LIMBS, LIMB_BITS>
{
    fn width(&self) -> usize {
        0
    }
}

impl<AB: InteractionBuilder + AirBuilder, const NUM_LIMBS: usize, const LIMB_BITS: usize> Air<AB>
    for ShiftAir<NUM_LIMBS, LIMB_BITS>
{
    fn eval(&self, _builder: &mut AB) {}
}
