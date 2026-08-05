use std::{array::from_fn, borrow::Borrow};

use openvm_circuit::{
    arch::U16_CELLS_PER_PUBLIC_VALUE,
    system::public_values::{public_values_event_block_from_limbs, public_values_initial_commit},
};
use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_continuations::{
    circuit::root::bus::{UserPvsCommitBus, UserPvsCommitMessage},
    utils::digests_to_poseidon2_input,
};
use openvm_recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    bus::{OutputValBus, OutputValMessage},
    output::VALS_IN_DIGEST,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct UserPvsCommitValuesCols<F> {
    pub is_valid: F,
    pub is_last: F,
    pub row_idx: F,
    pub len: F,
    pub value: [F; U16_CELLS_PER_PUBLIC_VALUE],
    pub commit_before: [F; DIGEST_SIZE],
    pub commit_after: [F; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(UserPvsCommitValuesCols<u8>)]
pub struct UserPvsCommitValuesAir {
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub user_pvs_commit_bus: UserPvsCommitBus,
    pub output_val_bus: OutputValBus,
    num_user_pvs: usize,
}

impl UserPvsCommitValuesAir {
    pub fn new(
        poseidon2_compress_bus: Poseidon2CompressBus,
        user_pvs_commit_bus: UserPvsCommitBus,
        output_val_bus: OutputValBus,
        num_user_pvs: usize,
    ) -> Self {
        assert!(num_user_pvs.is_multiple_of(U16_CELLS_PER_PUBLIC_VALUE));
        assert!((num_user_pvs / U16_CELLS_PER_PUBLIC_VALUE).is_power_of_two());
        Self {
            poseidon2_compress_bus,
            user_pvs_commit_bus,
            output_val_bus,
            num_user_pvs,
        }
    }
}

impl<F> BaseAir<F> for UserPvsCommitValuesAir {
    fn width(&self) -> usize {
        UserPvsCommitValuesCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for UserPvsCommitValuesAir {}
impl<F> PartitionedBaseAir<F> for UserPvsCommitValuesAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for UserPvsCommitValuesAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &UserPvsCommitValuesCols<AB::Var> = (*local).borrow();
        let next: &UserPvsCommitValuesCols<AB::Var> = (*next).borrow();

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
        for &value in &local.value {
            builder.when(not(local.is_valid)).assert_zero(value);
        }

        const OUTPUT_PUBLIC_VALUES_LEN_IDX: usize = (2 * DIGEST_SIZE) / VALS_IN_DIGEST;
        self.output_val_bus.send(
            builder,
            OutputValMessage {
                values: [local.len + local.is_valid, AB::Expr::ZERO],
                idx: AB::Expr::from_usize(OUTPUT_PUBLIC_VALUES_LEN_IDX),
            },
            local.is_last,
        );

        const OUTPUT_USER_PVS_START_IDX: usize = OUTPUT_PUBLIC_VALUES_LEN_IDX + 1;
        const OUTPUT_VAL_MSGS_PER_ROW: usize = U16_CELLS_PER_PUBLIC_VALUE / VALS_IN_DIGEST;
        for (i, values) in local.value.chunks_exact(VALS_IN_DIGEST).enumerate() {
            self.output_val_bus.send(
                builder,
                OutputValMessage {
                    values: from_fn(|i| values[i].into()),
                    idx: AB::Expr::from_usize(OUTPUT_USER_PVS_START_IDX + i)
                        + local.row_idx * AB::Expr::from_usize(OUTPUT_VAL_MSGS_PER_ROW),
                },
                AB::F::ONE,
            );
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
