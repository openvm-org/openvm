use itertools::Itertools;
use openvm_circuit::system::poseidon2::PERIPHERY_POSEIDON2_CHUNK_SIZE;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::PrimeCharacteristicRing,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferralPoseidon2Bus(pub LookupBus);

impl DeferralPoseidon2Bus {
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[inline(always)]
    pub fn index(&self) -> BusIndex {
        self.0.index
    }

    #[must_use]
    pub fn compress<T>(
        &self,
        left: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        right: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        output: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    ) -> DeferralPoseidon2Interaction<T> {
        DeferralPoseidon2Interaction {
            left: left.map(Into::into),
            right: right.map(Into::into),
            output: output.map(Into::into),
            bus: self.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeferralPoseidon2Interaction<T> {
    pub left: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub right: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub output: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub bus: LookupBus,
}

impl<T: PrimeCharacteristicRing> DeferralPoseidon2Interaction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let key: [_; 3 * PERIPHERY_POSEIDON2_CHUNK_SIZE] = self
            .left
            .into_iter()
            .chain(self.right)
            .chain(self.output)
            .collect_array()
            .unwrap();
        self.bus.lookup_key(builder, key, count);
    }
}
