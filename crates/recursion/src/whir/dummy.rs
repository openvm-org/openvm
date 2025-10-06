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
        StackingClaimsBus, StackingClaimsMessage, StackingCommitmentsBus,
        StackingCommitmentsBusMessage, StackingSumcheckRandomnessBus,
        StackingSumcheckRandomnessMessage, StackingWidthBusMessage, StackingWidthsBus,
        TranscriptBus, TranscriptBusMessage, WhirModuleBus, WhirModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyWhirCols<T> {
    is_first: T,
    tidx: T,

    stacking_commitments_msg: StackingCommitmentsBusMessage<T>,
    has_stacking_commitments_msg: T,
    stacking_widths_bus_msg: StackingWidthBusMessage<T>,
    has_stacking_widths_bus_msg: T,
    stacking_randomenss_msg: StackingSumcheckRandomnessMessage<T>,
    has_stacking_randomness_msg: T,
    stacking_claims_msg: StackingClaimsMessage<T>,
    has_stacking_claims_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

// Temporary dummy AIR to represent this module.
pub struct DummyWhirAir {
    pub whir_module_bus: WhirModuleBus,
    pub stacking_widths_bus: StackingWidthsBus,
    pub stacking_claims_bus: StackingClaimsBus,
    pub stacking_commitments_bus: StackingCommitmentsBus,
    pub stacking_randomness_bus: StackingSumcheckRandomnessBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for DummyWhirAir {}
impl PartitionedBaseAir<F> for DummyWhirAir {}

impl<F> BaseAir<F> for DummyWhirAir {
    fn width(&self) -> usize {
        DummyWhirCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyWhirAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &DummyWhirCols<AB::Var> = (*local).borrow();

        self.stacking_commitments_bus.receive(
            builder,
            local.stacking_commitments_msg.clone(),
            local.has_stacking_commitments_msg,
        );
        self.stacking_widths_bus.receive(
            builder,
            local.stacking_widths_bus_msg.clone(),
            local.has_stacking_widths_bus_msg,
        );
        self.stacking_randomness_bus.receive(
            builder,
            local.stacking_randomenss_msg.clone(),
            local.has_stacking_randomness_msg,
        );
        self.stacking_claims_bus.receive(
            builder,
            local.stacking_claims_msg.clone(),
            local.has_stacking_claims_msg,
        );
        self.whir_module_bus.receive(
            builder,
            WhirModuleMessage {
                tidx: local.tidx.into(),
            },
            local.is_first,
        );
        self.transcript_bus.receive(
            builder,
            local.transcript_msg.clone(),
            local.has_transcript_msg,
        );
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let mut stacking_commitments_msgs = preflight.stacking_commitments_msgs(vk, proof).into_iter();
    let mut stacking_widths_bus_msgs = preflight.stacking_widths_bus_msgs(vk).into_iter();
    let mut stacking_randomness_msgs = preflight.stacking_randomness_msgs().into_iter();
    let mut stacking_claims_msgs = preflight.stacking_claims_msgs(proof).into_iter();
    let mut transcript_msgs = preflight
        .transcript_msgs(preflight.stacking.post_tidx, preflight.transcript.len())
        .into_iter();

    let num_valid_rows: usize = [
        stacking_widths_bus_msgs.len(),
        stacking_commitments_msgs.len(),
        stacking_randomness_msgs.len(),
        stacking_claims_msgs.len(),
        transcript_msgs.len(),
    ]
    .into_iter()
    .max()
    .unwrap();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyWhirCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyWhirCols<F> = row.borrow_mut();
        cols.is_first = F::from_bool(i == 0);
        cols.tidx = F::from_canonical_usize(preflight.stacking.post_tidx);

        if let Some(msg) = stacking_commitments_msgs.next() {
            cols.stacking_commitments_msg = msg;
            cols.has_stacking_commitments_msg = F::ONE;
        }
        if let Some(msg) = stacking_widths_bus_msgs.next() {
            cols.stacking_widths_bus_msg = msg;
            cols.has_stacking_widths_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_randomness_msgs.next() {
            cols.stacking_randomenss_msg = msg;
            cols.has_stacking_randomness_msg = F::ONE;
        }
        if let Some(msg) = stacking_claims_msgs.next() {
            cols.stacking_claims_msg = msg;
            cols.has_stacking_claims_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
