use std::borrow::Borrow;

use itertools::Itertools;
use openvm_circuit::{
    arch::U16_CELLS_PER_PUBLIC_VALUE,
    system::public_values::{public_values_event_block_from_limbs, public_values_initial_commit},
};
use openvm_circuit_primitives::{
    encoder::Encoder,
    utils::{assert_array_eq, not},
    ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    circuit::root::bus::{UserPvsCommitBus, UserPvsCommitMessage},
    utils::digests_to_poseidon2_input,
};

pub(super) const MAX_ENCODER_DEGREE: u32 = 3;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct UserPvsCommitCols<F> {
    pub is_valid: F,
    pub is_last: F,
    pub row_idx: F,
    /// Number of values committed before this row.
    pub len: F,
    pub value: [F; U16_CELLS_PER_PUBLIC_VALUE],
    pub commit_before: [F; DIGEST_SIZE],
    pub commit_after: [F; DIGEST_SIZE],
}

/// Authenticates the fixed, zero-padded public-values output against the append-only commitment
/// carried by the final recursive VM proof.
pub struct UserPvsCommitAir {
    poseidon2_compress_bus: Poseidon2CompressBus,
    user_pvs_commit_bus: UserPvsCommitBus,
    encoder: Encoder,
    num_user_pvs: usize,
}

// The encoder columns are dynamic because their count depends on `num_user_pvs`.
impl ColumnsAir for UserPvsCommitAir {}

impl UserPvsCommitAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        num_user_pvs: usize,
    ) -> Self {
        assert!(num_user_pvs.is_multiple_of(U16_CELLS_PER_PUBLIC_VALUE));
        let num_values = num_user_pvs / U16_CELLS_PER_PUBLIC_VALUE;
        assert!(num_values.is_power_of_two());
        Self {
            poseidon2_compress_bus,
            user_pvs_commit_bus,
            encoder: Encoder::new(num_values, MAX_ENCODER_DEGREE, true),
            num_user_pvs,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitAir {
    fn width(&self) -> usize {
        UserPvsCommitCols::<u8>::width() + self.encoder.width()
    }
}

impl<F> BaseAirWithPublicValues<F> for UserPvsCommitAir {
    fn num_public_values(&self) -> usize {
        1 + self.num_user_pvs
    }
}

impl<F> PartitionedBaseAir<F> for UserPvsCommitAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB>
    for UserPvsCommitAir
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let cols_width = UserPvsCommitCols::<u8>::width();
        let (local, local_flags) = local.split_at(cols_width);
        let (next, _) = next.split_at(cols_width);
        let local: &UserPvsCommitCols<AB::Var> = (*local).borrow();
        let next: &UserPvsCommitCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_last);
        builder.when_transition().assert_zero(local.is_last);
        builder.when_last_row().assert_one(local.is_last);
        builder
            .when_transition()
            .assert_zero(next.is_valid * not(local.is_valid));
        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_eq(next.row_idx, local.row_idx + AB::Expr::ONE);

        builder.when_first_row().assert_zero(local.len);
        assert_array_eq(
            &mut builder.when_first_row(),
            local.commit_before,
            public_values_initial_commit::<AB::Expr>(
                self.num_user_pvs / U16_CELLS_PER_PUBLIC_VALUE,
            ),
        );
        builder
            .when_transition()
            .assert_eq(next.len, local.len + local.is_valid);
        assert_array_eq(
            &mut builder.when_transition(),
            next.commit_before,
            local.commit_after,
        );

        let event = public_values_event_block_from_limbs(local.value.map(Into::into));
        self.poseidon2_compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.commit_before.map(Into::into), event),
                output: local.commit_after.map(Into::into),
            },
            local.is_valid,
        );
        assert_array_eq(
            &mut builder.when(not(local.is_valid)),
            local.commit_after,
            local.commit_before,
        );

        debug_assert_eq!(self.encoder.width(), local_flags.len());
        self.encoder.eval(builder, local_flags);
        builder.assert_one(self.encoder.is_valid::<AB>(local_flags));

        let public_values = builder.public_values().iter().copied().collect_vec();
        let (num_values, pvs) = public_values.split_first().unwrap();
        builder
            .when_last_row()
            .assert_eq(*num_values, local.len + local.is_valid);
        let pvs = pvs.iter().copied().collect_vec();
        let mut selected_value = [AB::Expr::ZERO; U16_CELLS_PER_PUBLIC_VALUE];
        for (value_idx, value) in pvs.chunks_exact(U16_CELLS_PER_PUBLIC_VALUE).enumerate() {
            let selected = self.encoder.get_flag_expr::<AB>(value_idx, local_flags);
            builder
                .when(selected.clone())
                .assert_eq(local.row_idx, AB::Expr::from_usize(value_idx));
            for (dst, &cell) in selected_value.iter_mut().zip(value) {
                *dst += selected.clone() * Into::<AB::Expr>::into(cell);
            }
        }
        assert_array_eq(builder, local.value, selected_value);
        for &value in &local.value {
            builder.when(not(local.is_valid)).assert_zero(value);
        }

        self.user_pvs_commit_bus.receive(
            builder,
            UserPvsCommitMessage {
                commit: local.commit_after.map(Into::into),
            },
            local.is_last,
        );
    }
}
