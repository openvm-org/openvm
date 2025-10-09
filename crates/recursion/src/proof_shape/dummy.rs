use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    F, keygen::types::MultiStarkVerifyingKeyV2, poseidon2::sponge::FiatShamirTranscript,
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, GkrModuleBus,
        GkrModuleMessage, PublicValuesBus, StackingCommitmentsBus, StackingCommitmentsBusMessage,
        StackingWidthBusMessage, StackingWidthsBus, TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct DummyProofShapeCols<T> {
    gkr_bus_message: GkrModuleMessage<T>,
    has_gkr_bus_message: T,
    air_shape_bus_msg: AirShapeBusMessage<T>,
    has_air_shape_bus_msg: T,
    air_part_shape_bus_msg: AirPartShapeBusMessage<T>,
    has_air_part_shape_bus_msg: T,
    stacking_widths_bus_msg: StackingWidthBusMessage<T>,
    has_stacking_widths_bus_msg: T,
    stacking_commitments_msg: StackingCommitmentsBusMessage<T>,
    has_stacking_commitments_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

pub(crate) struct DummyProofShapeAir {
    pub transcript_bus: TranscriptBus,
    pub gkr_bus: GkrModuleBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_commitments_bus: StackingCommitmentsBus,
    pub stacking_widths_bus: StackingWidthsBus,
    pub _public_values_bus: PublicValuesBus,
}

impl<F> BaseAir<F> for DummyProofShapeAir {
    fn width(&self) -> usize {
        DummyProofShapeCols::<usize>::width()
    }
}

impl BaseAirWithPublicValues<F> for DummyProofShapeAir {}
impl PartitionedBaseAir<F> for DummyProofShapeAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyProofShapeAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyProofShapeCols<AB::Var> = (*local).borrow();
        let _next: &DummyProofShapeCols<AB::Var> = (*next).borrow();

        self.gkr_bus.send(
            builder,
            local.gkr_bus_message.clone(),
            local.has_gkr_bus_message,
        );

        self.air_shape_bus.send(
            builder,
            local.air_shape_bus_msg.clone(),
            AB::Expr::TWO * local.has_air_shape_bus_msg,
        );
        // TODO: We should probably send common main with AirShapeBus and only use this for
        // preprocessed/cached?
        self.air_part_shape_bus.send(
            builder,
            local.air_part_shape_bus_msg.clone(),
            AB::Expr::TWO * local.has_air_part_shape_bus_msg,
        );
        self.stacking_commitments_bus.send(
            builder,
            local.stacking_commitments_msg.clone(),
            local.has_stacking_commitments_msg,
        );
        self.stacking_widths_bus.send(
            builder,
            local.stacking_widths_bus_msg.clone(),
            AB::Expr::TWO * local.has_stacking_widths_bus_msg,
        );

        self.transcript_bus.receive(
            builder,
            local.transcript_msg.clone(),
            local.has_transcript_msg,
        );

        // TODO: public values
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let mut air_shape_bus_msgs = preflight.air_bus_msgs(vk).into_iter();
    let mut air_part_shape_bus_msgs = preflight.air_part_bus_msgs(vk).into_iter();
    let mut stacking_commitments_msgs = preflight.stacking_commitments_msgs(vk, proof).into_iter();
    let mut stacking_widths_bus_msgs = preflight.stacking_widths_bus_msgs(vk).into_iter();
    let mut gkr_bus_msgs = vec![GkrModuleMessage {
        tidx: F::from_canonical_usize(preflight.proof_shape.post_tidx),
        n_logup: F::from_canonical_usize(preflight.proof_shape.n_logup),
        n_max: F::from_canonical_usize(preflight.proof_shape.n_max),
    }]
    .into_iter();
    let mut transcript_msgs = preflight
        .transcript_msgs(0, preflight.proof_shape.post_tidx)
        .into_iter();

    let num_valid_rows = [
        air_shape_bus_msgs.len(),
        air_part_shape_bus_msgs.len(),
        stacking_commitments_msgs.len(),
        stacking_widths_bus_msgs.len(),
        gkr_bus_msgs.len(),
        transcript_msgs.len(),
    ]
    .into_iter()
    .max()
    .unwrap();

    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyProofShapeCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for row in trace.chunks_mut(width).take(num_valid_rows) {
        let cols: &mut DummyProofShapeCols<F> = row.borrow_mut();
        if let Some(msg) = gkr_bus_msgs.next() {
            cols.gkr_bus_message = msg;
            cols.has_gkr_bus_message = F::ONE;
        }
        if let Some(msg) = air_shape_bus_msgs.next() {
            cols.air_shape_bus_msg = msg;
            cols.has_air_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = air_part_shape_bus_msgs.next() {
            cols.air_part_shape_bus_msg = msg;
            cols.has_air_part_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_widths_bus_msgs.next() {
            cols.stacking_widths_bus_msg = msg;
            cols.has_stacking_widths_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_commitments_msgs.next() {
            cols.stacking_commitments_msg = msg;
            cols.has_stacking_commitments_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
