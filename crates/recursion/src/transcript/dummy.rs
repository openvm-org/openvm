use core::{
    borrow::{Borrow, BorrowMut},
    cmp::max,
};

use itertools::Itertools;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, keygen::types::MultiStarkVerifyingKeyV2, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{CommitmentsBus, CommitmentsBusMessage, TranscriptBus, TranscriptBusMessage},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct DummyTranscriptCols<T> {
    proof_idx: T,
    has_transcript_msg: T,
    transcript_msg: TranscriptBusMessage<T>,
    has_commitment_msg: T,
    commitment_msg: CommitmentsBusMessage<T>,
}

pub struct DummyTranscriptAir {
    pub transcript_bus: TranscriptBus,
    pub commitments_bus: CommitmentsBus,
}

impl BaseAirWithPublicValues<F> for DummyTranscriptAir {}
impl PartitionedBaseAir<F> for DummyTranscriptAir {}

impl<F> BaseAir<F> for DummyTranscriptAir {
    fn width(&self) -> usize {
        DummyTranscriptCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for DummyTranscriptAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &DummyTranscriptCols<AB::Var> = (*local).borrow();

        self.transcript_bus.send(
            builder,
            local.proof_idx,
            local.transcript_msg.clone(),
            local.has_transcript_msg,
        );
        self.commitments_bus.receive(
            builder,
            local.proof_idx,
            local.commitment_msg.clone(),
            local.has_commitment_msg,
        );
    }
}

pub(crate) fn generate_trace(
    vk: &MultiStarkVerifyingKeyV2,
    proof: &Proof,
    preflight: &Preflight,
) -> RowMajorMatrix<F> {
    let commit_msgs = preflight
        .stacking_commitments_msgs(vk, proof)
        .into_iter()
        .chain(preflight.whir_commitments_msgs(proof))
        .collect_vec();
    let transcript_len: usize = preflight.transcript.len();

    let num_valid_rows = max(commit_msgs.len(), transcript_len);
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyTranscriptCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyTranscriptCols<F> = row.borrow_mut();

        cols.has_transcript_msg = F::from_bool(i < transcript_len);
        cols.transcript_msg = TranscriptBusMessage {
            tidx: F::from_canonical_usize(i),
            value: preflight.transcript[i],
            is_sample: F::from_bool(preflight.transcript.samples()[i]),
        };
        if i < commit_msgs.len() {
            cols.has_commitment_msg = F::ONE;
            cols.commitment_msg = commit_msgs[i].clone();
        }
    }

    RowMajorMatrix::new(trace, width)
}
