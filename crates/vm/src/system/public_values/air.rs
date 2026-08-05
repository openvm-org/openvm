use std::borrow::Borrow;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::VM_DIGEST_WIDTH;
use openvm_stark_backend::{
    interaction::{InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use super::{
    assert_public_values_shape, public_values_event_block_from_limbs, public_values_trace_height,
    PublicValuesBus, PUBLIC_VALUE_LIMBS,
};

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct PublicValuesCols<T> {
    /// Whether this row corresponds to a reveal in the current segment.
    pub is_valid: T,
    /// Segment-local reveal ordinal, equal to the physical row index.
    pub ordinal: T,
    /// Accumulator before this row.
    pub commit: [T; VM_DIGEST_WIDTH],
    /// Poseidon2 compression of `commit` with this row's event block.
    pub hash: [T; VM_DIGEST_WIDTH],
    /// Revealed `u64`, encoded as four little-endian `u16` limbs.
    pub value: [T; PUBLIC_VALUE_LIMBS],
}

#[derive(Debug, Clone, Copy, AlignedBorrow, StructReflection)]
#[repr(C)]
pub struct PublicValuesPvs<T> {
    pub initial_commit: [T; VM_DIGEST_WIDTH],
    pub final_commit: [T; VM_DIGEST_WIDTH],
}

/// Proves one append-only public-output accumulator transition.
#[derive(Clone, Debug, ColumnsAir)]
#[columns_via(PublicValuesCols<u8>)]
pub struct PublicValuesAir {
    pub num_public_value_cells: usize,
    pub public_values_bus: PublicValuesBus,
    pub compression_bus: PermutationCheckBus,
}

impl PublicValuesAir {
    pub fn new(
        num_public_value_cells: usize,
        public_values_bus: PublicValuesBus,
        compression_bus: PermutationCheckBus,
    ) -> Self {
        assert_public_values_shape(num_public_value_cells);
        Self {
            num_public_value_cells,
            public_values_bus,
            compression_bus,
        }
    }

    pub fn trace_height(&self) -> usize {
        public_values_trace_height(self.num_public_value_cells)
    }
}

impl<F: Field> BaseAir<F> for PublicValuesAir {
    fn width(&self) -> usize {
        PublicValuesCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for PublicValuesAir {
    fn num_public_values(&self) -> usize {
        PublicValuesPvs::<F>::width()
    }
}

impl<F: Field> PartitionedBaseAir<F> for PublicValuesAir {}

impl<AB: InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for PublicValuesAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).unwrap();
        let next_row = main.row_slice(1).unwrap();
        let local: &PublicValuesCols<_> = (*local_row).borrow();
        let next: &PublicValuesCols<_> = (*next_row).borrow();

        let &PublicValuesPvs {
            initial_commit,
            final_commit,
        } = builder.public_values().borrow();

        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .when(AB::Expr::ONE - local.is_valid)
            .assert_zero(next.is_valid);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when_transition()
            .assert_eq(next.ordinal, local.ordinal + AB::Expr::ONE);

        for i in 0..VM_DIGEST_WIDTH {
            builder
                .when_first_row()
                .assert_eq(local.commit[i], initial_commit[i]);
            builder.when_transition().assert_eq(
                next.commit[i],
                local.commit[i] + local.is_valid * (local.hash[i] - local.commit[i]),
            );
            builder.when_last_row().assert_eq(
                final_commit[i],
                local.commit[i] + local.is_valid * (local.hash[i] - local.commit[i]),
            );
        }

        for limb in local.value {
            builder
                .when(AB::Expr::ONE - local.is_valid)
                .assert_zero(limb);
        }

        let event = public_values_event_block_from_limbs(local.value.map(Into::into));
        self.compression_bus.interact(
            builder,
            local
                .commit
                .into_iter()
                .map(Into::into)
                .chain(event)
                .chain(local.hash.into_iter().map(Into::into)),
            local.is_valid,
        );

        self.public_values_bus
            .receive(local.ordinal, local.value)
            .eval(builder, local.is_valid);
    }
}
