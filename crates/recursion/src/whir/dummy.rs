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
        CommitmentsBus, CommitmentsBusMessage, StackingIndexMessage, StackingIndicesBus,
        StackingSumcheckRandomnessBus, StackingSumcheckRandomnessMessage, TranscriptBus,
        TranscriptBusMessage, WhirModuleBus, WhirModuleMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct DummyWhirCols<T> {
    is_first: T,
    proof_idx: T,
    whir_module_msg: WhirModuleMessage<T>,
    commitments_msg: CommitmentsBusMessage<T>,
    has_commitments_msg: T,
    stacking_widths_bus_msg: StackingIndexMessage<T>,
    has_stacking_widths_bus_msg: T,
    stacking_randomenss_msg: StackingSumcheckRandomnessMessage<T>,
    has_stacking_randomness_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

// Temporary dummy AIR to represent this module.
pub struct DummyWhirAir {
    pub whir_module_bus: WhirModuleBus,
    pub stacking_widths_bus: StackingIndicesBus,
    pub commitments_bus: CommitmentsBus,
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

        self.commitments_bus.send(
            builder,
            local.proof_idx,
            local.commitments_msg.clone(),
            local.has_commitments_msg,
        );
        self.stacking_widths_bus.receive(
            builder,
            local.proof_idx,
            local.stacking_widths_bus_msg.clone(),
            local.has_stacking_widths_bus_msg,
        );
        self.stacking_randomness_bus.receive(
            builder,
            local.proof_idx,
            local.stacking_randomenss_msg.clone(),
            local.has_stacking_randomness_msg,
        );
        self.whir_module_bus.receive(
            builder,
            local.proof_idx,
            local.whir_module_msg.clone(),
            local.is_first,
        );
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
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
    let mut commitments_msgs = preflight.whir_commitments_msgs(proof).into_iter();
    let mut stacking_widths_bus_msgs = preflight.stacking_widths_bus_msgs(vk).into_iter();
    let mut stacking_randomness_msgs = preflight.stacking_randomness_msgs().into_iter();
    let mut transcript_msgs = preflight
        .transcript_msgs(preflight.stacking.post_tidx, preflight.transcript.len())
        .into_iter();

    let num_valid_rows: usize = [
        stacking_widths_bus_msgs.len(),
        commitments_msgs.len(),
        stacking_randomness_msgs.len(),
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

        if i == 0 {
            cols.whir_module_msg = preflight.whir_module_msg(proof);
        }

        if let Some(msg) = commitments_msgs.next() {
            cols.commitments_msg = msg;
            cols.has_commitments_msg = F::ONE;
        }
        if let Some(msg) = stacking_widths_bus_msgs.next() {
            cols.stacking_widths_bus_msg = msg;
            cols.has_stacking_widths_bus_msg = F::ONE;
        }
        if let Some(msg) = stacking_randomness_msgs.next() {
            cols.stacking_randomenss_msg = msg;
            cols.has_stacking_randomness_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
