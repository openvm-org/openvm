use std::{cmp::max, iter::once, marker::PhantomData};

use ndarray::s;
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    encoder::Encoder,
    utils::{not, select},
    SubAir,
};
use openvm_sha2_air::{compose, Sha2BlockHasherSubAir, Sha2DigestColsRef, Sha2RoundColsRef};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{
    MessageType, Sha2BlockHasherDigestColsRef, Sha2BlockHasherRoundColsRef,
    Sha2BlockHasherVmConfig, INNER_OFFSET,
};

pub struct Sha2BlockHasherVmAir<C: Sha2BlockHasherVmConfig> {
    pub inner: Sha2BlockHasherSubAir<C>,
    pub sha2_bus: PermutationCheckBus,
}

impl<C: Sha2BlockHasherVmConfig> Sha2BlockHasherVmAir<C> {
    pub fn new(
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        inner_bus_idx: BusIndex,
        sha2_bus_idx: BusIndex,
    ) -> Self {
        Self {
            inner: Sha2BlockHasherSubAir::new(bitwise_lookup_bus, inner_bus_idx),
            sha2_bus: PermutationCheckBus::new(sha2_bus_idx),
        }
    }
}

impl<F: Field, C: Sha2BlockHasherVmConfig> BaseAirWithPublicValues<F> for Sha2BlockHasherVmAir<C> {}
impl<F: Field, C: Sha2BlockHasherVmConfig> PartitionedBaseAir<F> for Sha2BlockHasherVmAir<C> {}
impl<F: Field, C: Sha2BlockHasherVmConfig> BaseAir<F> for Sha2BlockHasherVmAir<C> {
    fn width(&self) -> usize {
        C::BLOCK_HASHER_WIDTH
    }
}

impl<AB: InteractionBuilder, C: Sha2BlockHasherVmConfig> Air<AB> for Sha2BlockHasherVmAir<C> {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder, INNER_OFFSET);
        self.eval_interactions(builder);
        self.eval_request_id(builder);
    }
}

impl<C: Sha2BlockHasherVmConfig> Sha2BlockHasherVmAir<C> {
    fn eval_interactions<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local_slice = main.row_slice(0);
        let next_slice = main.row_slice(1);

        let local = Sha2BlockHasherDigestColsRef::<AB::Var>::from::<C>(
            &local_slice[..C::BLOCK_HASHER_DIGEST_WIDTH],
        );

        // Receive (STATE, request_id, prev_state_as_u16s, new_state) on the sha2 bus
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::State as u8),
                (*local.request_id).into(),
            ]
            .into_iter()
            .chain(local.inner.prev_hash.flatten().map(|x| (*x).into()))
            .chain(local.inner.final_hash.flatten().map(|x| (*x).into())),
            *local.inner.flags.is_digest_row,
        );

        let local = Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(
            &local_slice[..C::BLOCK_HASHER_ROUND_WIDTH],
        );
        let next = Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(
            &next_slice[..C::BLOCK_HASHER_ROUND_WIDTH],
        );

        let is_local_first_row = self
            .inner
            .row_idx_encoder
            .contains_flag::<AB>(local.inner.flags.row_idx.to_slice().unwrap(), &[0]);

        // Taken from old Sha256VmChip:
        // https://github.com/openvm-org/openvm/blob/c2e376e6059c8bbf206736cf01d04cda43dfc42d/extensions/sha256/circuit/src/sha256_chip/air.rs#L310C1-L318C1
        let get_ith_byte = |i: usize, cols: &Sha2BlockHasherRoundColsRef<AB::Var>| {
            debug_assert!(i < C::WORD_U8S * C::ROUNDS_PER_ROW);
            let row_idx = i / C::WORD_U8S;
            let word: Vec<AB::Var> = cols
                .inner
                .message_schedule
                .w
                .row(row_idx)
                .into_iter()
                .copied()
                .collect::<Vec<_>>();
            // Need to reverse the byte order to match the endianness of the memory
            let byte_idx = C::WORD_U8S - i % C::WORD_U8S - 1;
            compose::<AB::Expr>(&word[byte_idx * 8..(byte_idx + 1) * 8], 1)
        };

        let local_message = (0..C::WORD_U8S * C::ROUNDS_PER_ROW).map(|i| get_ith_byte(i, &local));
        let next_message = (0..C::WORD_U8S * C::ROUNDS_PER_ROW).map(|i| get_ith_byte(i, &next));

        // Receive (MESSAGE_1, request_id, first_half_of_message) on the sha2 bus
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::Message1 as u8),
                (*local.request_id).into(),
            ]
            .into_iter()
            .chain(local_message.clone())
            .chain(next_message.clone()),
            is_local_first_row * local.inner.flags.is_not_padding_row(),
        );

        let is_local_third_row = self
            .inner
            .row_idx_encoder
            .contains_flag::<AB>(local.inner.flags.row_idx.to_slice().unwrap(), &[2]);

        // Send (MESSAGE_2, request_id, second_half_of_message) to the sha2 bus
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_canonical_u8(MessageType::Message2 as u8),
                (*local.request_id).into(),
            ]
            .into_iter()
            .chain(local_message)
            .chain(next_message),
            is_local_third_row * local.inner.flags.is_not_padding_row(),
        );
    }

    fn eval_request_id<AB: InteractionBuilder>(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let next = main.row_slice(1);

        // doesn't matter if we use round or digest cols here, since we only access
        // request_id and inner.flags.is_last block, which are common to both
        // field
        let local =
            Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(&local[..C::BLOCK_HASHER_WIDTH]);
        let next =
            Sha2BlockHasherRoundColsRef::<AB::Var>::from::<C>(&next[..C::BLOCK_HASHER_WIDTH]);

        builder
            .when_transition()
            .when(*local.inner.flags.is_round_row)
            .assert_eq(*next.request_id, *local.request_id);
    }
}
