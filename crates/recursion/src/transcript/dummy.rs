use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{F, poseidon2::sponge::FiatShamirTranscript, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{TranscriptBus, TranscriptBusMessage},
    system::Preflight,
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct DummyTranscriptCols<T> {
    is_valid: T,
    msg: TranscriptBusMessage<T>,
}

pub struct DummyTranscriptAir {
    pub transcript_bus: TranscriptBus,
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

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &DummyTranscriptCols<AB::Var> = (*local).borrow();
        let next: &DummyTranscriptCols<AB::Var> = (*next).borrow();

        self.transcript_bus
            .send(builder, local.msg.clone(), local.is_valid);
    }
}

pub(crate) fn generate_trace<TS: FiatShamirTranscript>(
    proof: &Proof,
    preflight: &Preflight<TS>,
) -> RowMajorMatrix<F> {
    let num_valid_rows: usize = preflight.transcript.len();
    let num_rows = num_valid_rows.next_power_of_two();
    let width = DummyTranscriptCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut DummyTranscriptCols<F> = row.borrow_mut();

        cols.is_valid = F::ONE;
        cols.msg = TranscriptBusMessage {
            tidx: F::from_canonical_usize(i),
            value: preflight.transcript.data[i],
            is_sample: F::from_bool(preflight.transcript.is_sample[i]),
        };
    }

    RowMajorMatrix::new(trace, width)
}
