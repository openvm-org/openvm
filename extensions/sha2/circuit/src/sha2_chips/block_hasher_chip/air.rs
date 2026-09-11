use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus, ColumnsAir,
    SubAir,
};
use openvm_sha2_air::{compose, Sha2BlockHasherSubAir};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::{
    MessageType, Sha2BlockHasherDigestColsRef, Sha2BlockHasherRoundColsRef,
    Sha2BlockHasherVmConfig, INNER_OFFSET,
};

pub struct Sha2BlockHasherVmAir<C: Sha2BlockHasherVmConfig> {
    pub inner: Sha2BlockHasherSubAir<C>,
    pub sha2_bus: PermutationCheckBus,
}

// No columns provided: width is the config-dependent `C::BLOCK_HASHER_WIDTH` and rows are accessed
// via `Sha2BlockHasher{Round,Digest}ColsRef` (slice-borrowing ref structs, no static `Cols`).
impl<C: Sha2BlockHasherVmConfig> ColumnsAir for Sha2BlockHasherVmAir<C> {}

impl<C: Sha2BlockHasherVmConfig> Sha2BlockHasherVmAir<C> {
    pub fn new(
        bitwise_lookup_bus: BitwiseOperationLookupBus,
        range_bus: VariableRangeCheckerBus,
        inner_bus_idx: BusIndex,
        sha2_bus_idx: BusIndex,
    ) -> Self {
        Self {
            inner: Sha2BlockHasherSubAir::new(bitwise_lookup_bus, range_bus, inner_bus_idx),
            sha2_bus: PermutationCheckBus::new(sha2_bus_idx),
        }
    }

    /// Read one message byte from the SHA-2 message schedule columns.
    fn message_byte<AB: InteractionBuilder>(
        i: usize,
        cols: &Sha2BlockHasherRoundColsRef<AB::Var>,
    ) -> AB::Expr {
        let round_row_u8s = C::WORD_U8S * C::ROUNDS_PER_ROW;
        debug_assert!(i < round_row_u8s);
        let row_idx = i / C::WORD_U8S;
        let word = cols.inner.message_schedule.w.row(row_idx);
        // Reverse byte order to match memory endianness.
        let byte_idx = C::WORD_U8S - i % C::WORD_U8S - 1;
        let byte_bits = u8::BITS as usize;
        let bit_range = byte_idx * byte_bits..(byte_idx + 1) * byte_bits;
        compose::<AB::Expr>(&word.as_slice().unwrap()[bit_range], 1)
    }

    /// Read one little-endian u16 cell from two message bytes.
    fn message_u16<AB: InteractionBuilder>(
        k: usize,
        cols: &Sha2BlockHasherRoundColsRef<AB::Var>,
    ) -> AB::Expr {
        let byte_idx = 2 * k;
        // `message_byte` returns memory-order bytes; pack adjacent bytes as one LE u16 cell.
        Self::message_byte::<AB>(byte_idx, cols)
            + AB::Expr::from_u32(1u32 << u8::BITS) * Self::message_byte::<AB>(byte_idx + 1, cols)
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
        let local_slice = main.row_slice(0).unwrap();
        let next_slice = main.row_slice(1).unwrap();

        let local = Sha2BlockHasherDigestColsRef::<AB::Var>::from::<C>(
            &local_slice[..C::BLOCK_HASHER_DIGEST_WIDTH],
        );

        // Receive (STATE, request_id, prev_state, new_state) on the SHA-2 bus.
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_u8(MessageType::State as u8),
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

        let round_row_u16s = C::WORD_U16S * C::ROUNDS_PER_ROW;
        let local_message = (0..round_row_u16s).map(|k| Self::message_u16::<AB>(k, &local));
        let next_message = (0..round_row_u16s).map(|k| Self::message_u16::<AB>(k, &next));

        // Receive (MESSAGE_1, request_id, first half of message) on the SHA-2 bus.
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_u8(MessageType::Message1 as u8),
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

        // Receive (MESSAGE_2, request_id, second half of message) on the SHA-2 bus.
        self.sha2_bus.receive(
            builder,
            [
                AB::Expr::from_u8(MessageType::Message2 as u8),
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
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();

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
