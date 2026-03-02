use std::borrow::Borrow;

use openvm_circuit_primitives::utils::not;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_matrix::Matrix;
use recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    prelude::DIGEST_SIZE,
    utils::assert_zeros,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    circuit::deferral::aggregation::root::bus::{
        DefVkCommitBus, DefVkCommitMessage, IoCommitBus, IoCommitMessage, OnionResultBus,
        OnionResultMessage,
    },
    utils::digests_to_poseidon2_input,
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct OnionHashCols<F> {
    pub row_idx: F,
    pub is_valid: F,
    pub is_first: F,

    pub input_commit: [F; DIGEST_SIZE],
    pub output_commit: [F; DIGEST_SIZE],

    pub input_onion: [F; DIGEST_SIZE],
    pub output_onion: [F; DIGEST_SIZE],
}

pub struct OnionHashAir {
    pub poseidon2_bus: Poseidon2CompressBus,
    pub def_vk_commit_bus: DefVkCommitBus,
    pub io_commit_bus: IoCommitBus,
    pub onion_res_bus: OnionResultBus,
}

impl<F> BaseAir<F> for OnionHashAir {
    fn width(&self) -> usize {
        OnionHashCols::<u8>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for OnionHashAir {}
impl<F> PartitionedBaseAir<F> for OnionHashAir {}

impl<AB: AirBuilder + InteractionBuilder + AirBuilderWithPublicValues> Air<AB> for OnionHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &OnionHashCols<AB::Var> = (*local).borrow();
        let next: &OnionHashCols<AB::Var> = (*next).borrow();

        /*
         * Base constraints to ensure that all the valid rows are at the beginning,
         * and that there is at least one valid and one invalid row. The latter is
         * important because the final onion values are read from the first invalid row.
         */
        builder.assert_bool(local.is_valid);
        builder.when_first_row().assert_one(local.is_valid);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);
        builder.when_last_row().assert_zero(local.is_valid);

        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);

        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_one(next.row_idx - local.row_idx);

        /*
         * On the first row we want input_onion to initially be def_vk_commit and
         * output_onion to be all zeroes.
         */
        assert_zeros(&mut builder.when(local.is_first), local.output_onion);
        self.def_vk_commit_bus.receive(
            builder,
            DefVkCommitMessage {
                def_vk_commit: local.input_onion,
            },
            local.is_first,
        );

        /*
         * On valid rows we want to receive the input and output commit values and
         * hash them with the current input and output onions. We send the final
         * onion values on the transition from the last valid row to the first invalid row.
         */
        self.io_commit_bus.receive(
            builder,
            IoCommitMessage {
                idx: local.row_idx,
                input_commit: local.input_commit,
                output_commit: local.output_commit,
            },
            local.is_valid,
        );

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.input_onion, local.input_commit),
                output: next.input_onion,
            },
            local.is_valid,
        );

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: digests_to_poseidon2_input(local.output_onion, local.output_commit),
                output: next.output_onion,
            },
            local.is_valid,
        );

        self.onion_res_bus.send(
            builder,
            OnionResultMessage {
                input_onion: next.input_onion,
                output_onion: next.output_onion,
            },
            local.is_valid * not(next.is_valid),
        );
    }
}
