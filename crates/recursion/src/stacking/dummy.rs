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
        AirPartShapeBus, AirPartShapeBusMessage, AirShapeBus, AirShapeBusMessage, ColumnClaimsBus,
        ColumnClaimsMessage, ConstraintSumcheckRandomness, ConstraintSumcheckRandomnessBus,
        StackingIndexMessage, StackingIndicesBus, StackingModuleBus, StackingModuleMessage,
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, TranscriptBus,
        TranscriptBusMessage, WhirModuleBus, WhirModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyStackingCols<T> {
    is_first: T,
    stacking_tidx: T,
    whir_module_msg: WhirModuleMessage<T>,
    constraint_sumcheck_rnd_msg: ConstraintSumcheckRandomness<T>,
    has_constraint_sumcheck_rnd: T,
    column_claim_msg: ColumnClaimsMessage<T>,
    has_column_claim_msg: T,
    air_shape_bus_msg: AirShapeBusMessage<T>,
    has_air_shape_bus_msg: T,
    air_part_shape_bus_msg: AirPartShapeBusMessage<T>,
    has_air_part_shape_bus_msg: T,
    stacking_randomness_bus_msg: StackingSumcheckRandomnessMessage<T>,
    has_stacking_randomness_bus_msg: T,
    stacking_widths_bus_msg: StackingIndexMessage<T>,
    has_stacking_widths_bus_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

// Temporary dummy AIR to represent this module.
pub struct DummyStackingAir {
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub batch_constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
    pub air_shape_bus: AirShapeBus,
    pub air_part_shape_bus: AirPartShapeBus,
    pub stacking_widths_bus: StackingIndicesBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for DummyStackingAir {}
impl PartitionedBaseAir<F> for DummyStackingAir {}

impl<F> BaseAir<F> for DummyStackingAir {
    fn width(&self) -> usize {
        DummyStackingCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyStackingAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &DummyStackingCols<AB::Var> = (*local).borrow();

        self.stacking_module_bus.receive(
            builder,
            StackingModuleMessage {
                tidx: local.stacking_tidx.into(),
            },
            local.is_first,
        );
        self.batch_constraint_randomness_bus.receive(
            builder,
            local.constraint_sumcheck_rnd_msg.clone(),
            local.has_constraint_sumcheck_rnd,
        );
        self.stacking_randomness_bus.send(
            builder,
            local.stacking_randomness_bus_msg.clone(),
            local.has_stacking_randomness_bus_msg,
        );
        self.whir_module_bus
            .send(builder, local.whir_module_msg.clone(), local.is_first);
        self.column_claims_bus.receive(
            builder,
            local.column_claim_msg.clone(),
            local.has_column_claim_msg,
        );
        self.air_shape_bus.receive(
            builder,
            local.air_shape_bus_msg.clone(),
            local.has_air_shape_bus_msg,
        );
        self.air_part_shape_bus.receive(
            builder,
            local.air_part_shape_bus_msg.clone(),
            local.has_air_part_shape_bus_msg,
        );
        self.stacking_widths_bus.receive(
            builder,
            local.stacking_widths_bus_msg.clone(),
            local.has_stacking_widths_bus_msg,
        );
        self.transcript_bus.receive(
            builder,
            local.transcript_msg.clone(),
            local.has_transcript_msg,
        )
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let mut constraint_sc_msgs = preflight.batch_constraint_sumcheck_randomness().into_iter();
    let mut column_claims_bus_msgs = preflight.column_claims_messages(vk, proof).into_iter();
    let mut air_shape_bus_msgs = preflight.air_bus_msgs(vk).into_iter();
    let mut air_part_shape_bus_msgs = preflight.air_part_bus_msgs(vk).into_iter();
    let mut stacking_widths_bus_msgs = preflight.stacking_widths_bus_msgs(vk).into_iter();
    let mut stacking_randomness_msgs = preflight.stacking_randomness_msgs().into_iter();
    let mut transcript_msgs = preflight
        .transcript_msgs(
            preflight.batch_constraint.post_tidx,
            preflight.stacking.post_tidx,
        )
        .into_iter();

    let num_valid_rows = [
        constraint_sc_msgs.len(),
        column_claims_bus_msgs.len(),
        air_shape_bus_msgs.len(),
        air_part_shape_bus_msgs.len(),
        stacking_widths_bus_msgs.len(),
        transcript_msgs.len(),
    ]
    .into_iter()
    .max()
    .unwrap();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyStackingCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyStackingCols<F> = row.borrow_mut();
        cols.is_first = F::from_bool(i == 0);
        cols.stacking_tidx = F::from_canonical_usize(preflight.batch_constraint.post_tidx);

        cols.whir_module_msg = preflight.whir_module_msg(proof);
        if let Some(msg) = constraint_sc_msgs.next() {
            cols.constraint_sumcheck_rnd_msg = msg;
            cols.has_constraint_sumcheck_rnd = F::ONE;
        }
        if let Some(msg) = column_claims_bus_msgs.next() {
            cols.column_claim_msg = msg;
            cols.has_column_claim_msg = F::ONE;
        }
        if let Some(msg) = air_shape_bus_msgs.next() {
            cols.air_shape_bus_msg = msg;
            cols.has_air_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = air_part_shape_bus_msgs.next() {
            cols.air_part_shape_bus_msg = msg;
            cols.has_air_part_shape_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_randomness_msgs.next() {
            cols.stacking_randomness_bus_msg = msg;
            cols.has_stacking_randomness_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_widths_bus_msgs.next() {
            cols.stacking_widths_bus_msg = msg;
            cols.has_stacking_widths_bus_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
