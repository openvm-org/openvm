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
        BatchConstraintModuleBus, BatchConstraintModuleMessage, GkrModuleBus, GkrModuleMessage,
        TranscriptBus, TranscriptBusMessage, XiRandomnessBus, XiRandomnessMessage,
    },
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct DummyGkrRoundCols<T> {
    gkr_module_msg: GkrModuleMessage<T>,
    has_gkr_module_msg: T,
    batch_constraint_module_msg: BatchConstraintModuleMessage<T>,
    has_batch_constraint_module_msg: T,
    xi_randomness_msg: XiRandomnessMessage<T>,
    has_xi_ranodmness_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_transcript_msg: T,
}

pub struct DummyGkrRoundAir {
    pub gkr_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub xi_randomness_bus: XiRandomnessBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for DummyGkrRoundAir {}
impl PartitionedBaseAir<F> for DummyGkrRoundAir {}

impl<F> BaseAir<F> for DummyGkrRoundAir {
    fn width(&self) -> usize {
        DummyGkrRoundCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyGkrRoundAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &DummyGkrRoundCols<AB::Var> = (*local).borrow();

        self.gkr_bus.receive(
            builder,
            local.gkr_module_msg.clone(),
            local.has_gkr_module_msg,
        );
        self.bc_module_bus.send(
            builder,
            local.batch_constraint_module_msg.clone(),
            local.has_batch_constraint_module_msg,
        );
        self.xi_randomness_bus.send(
            builder,
            local.xi_randomness_msg.clone(),
            local.has_xi_ranodmness_msg,
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
    let n_logup = proof.gkr_proof.claims_per_layer.len();
    let n_max = preflight.proof_shape.n_max;

    let mut gkr_module_msgs = [GkrModuleMessage {
        tidx: F::from_canonical_usize(preflight.proof_shape.post_tidx),
        n_logup: F::from_canonical_usize(n_logup),
        n_max: F::from_canonical_usize(n_max),
    }]
    .into_iter();
    let mut batch_constraint_module_msgs =
        preflight.batch_constraint_module_msgs(proof).into_iter();
    let mut xi_randomness_msgs = preflight.xi_randomness_messages().into_iter();
    let mut transcript_msgs = preflight
        .transcript_msgs(preflight.proof_shape.post_tidx, preflight.gkr.post_tidx)
        .into_iter();

    let num_valid_rows = [
        gkr_module_msgs.len(),
        batch_constraint_module_msgs.len(),
        xi_randomness_msgs.len(),
        transcript_msgs.len(),
    ]
    .into_iter()
    .max()
    .unwrap();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyGkrRoundCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for row in trace.chunks_mut(width).take(num_valid_rows) {
        let cols: &mut DummyGkrRoundCols<F> = row.borrow_mut();
        if let Some(msg) = gkr_module_msgs.next() {
            cols.gkr_module_msg = msg;
            cols.has_gkr_module_msg = F::ONE;
        }
        if let Some(msg) = batch_constraint_module_msgs.next() {
            cols.batch_constraint_module_msg = msg;
            cols.has_batch_constraint_module_msg = F::ONE;
        }
        if let Some(msg) = xi_randomness_msgs.next() {
            cols.xi_randomness_msg = msg;
            cols.has_xi_ranodmness_msg = F::ONE;
        }
        if let Some(msg) = transcript_msgs.next() {
            cols.transcript_msg = msg;
            cols.has_transcript_msg = F::ONE;
        }
    }

    RowMajorMatrix::new(trace, width)
}
