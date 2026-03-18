use std::iter::once;

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
    pub fn lookup<T>(
        &self,
        left: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        right: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        output: [impl Into<T>; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        is_compress: impl Into<T>,
    ) -> DeferralPoseidon2Interaction<T> {
        DeferralPoseidon2Interaction {
            left: left.map(Into::into),
            right: right.map(Into::into),
            output: output.map(Into::into),
            is_compress: is_compress.into(),
            bus: self.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeferralPoseidon2Interaction<T> {
    pub left: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub right: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub output: [T; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    pub is_compress: T,
    pub bus: LookupBus,
}

impl<T: PrimeCharacteristicRing> DeferralPoseidon2Interaction<T> {
    pub fn eval<AB>(self, builder: &mut AB, count: impl Into<AB::Expr>)
    where
        AB: InteractionBuilder<Expr = T>,
    {
        let key: [_; 3 * PERIPHERY_POSEIDON2_CHUNK_SIZE + 1] = once(self.is_compress)
            .chain(self.left)
            .chain(self.right)
            .chain(self.output)
            .collect_array()
            .unwrap();
        self.bus.lookup_key(builder, key, count);
    }
}
