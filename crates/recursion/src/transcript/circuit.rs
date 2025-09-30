use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::FieldAlgebra;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use stark_backend_v2::{
    DIGEST_SIZE, F,
    poseidon2::{CHUNK, WIDTH},
    proof::Proof,
};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::bus::{TranscriptBus, TranscriptBusMessage};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
struct TranscriptCols<T> {
    tidx: T,
    mask: [T; CHUNK],
    preimage: [T; WIDTH],
    image: [T; WIDTH],
    is_sample: T,
}

pub struct TranscriptAir {
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for TranscriptAir {}
impl PartitionedBaseAir<F> for TranscriptAir {}

impl<F> BaseAir<F> for TranscriptAir {
    fn width(&self) -> usize {
        TranscriptCols::<usize>::width()
    }
}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for TranscriptAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TranscriptCols<AB::Var> = (*local).borrow();
        let next: &TranscriptCols<AB::Var> = (*next).borrow();

        let is_valid = local.mask[0];
        builder.when_first_row().assert_one(is_valid);
        builder.assert_bool(is_valid);
        builder
            .when_transition()
            .when(is_valid)
            .assert_one(is_valid);

        let mut skip = AB::Expr::ZERO;
        for i in 0..CHUNK {
            builder.assert_bool(local.mask[i]);
            if i > 0 {
                builder.when(local.mask[i]).assert_one(local.mask[i - 1]);
            }

            skip += local.mask[i].into();

            self.transcript_bus.send(
                builder,
                TranscriptBusMessage {
                    tidx: local.tidx.into() + AB::Expr::from_canonical_usize(i),
                    value: local.preimage[i].into(),
                    is_sample: local.is_sample.into(),
                },
                local.mask[i],
            );
        }

        builder
            .when_transition()
            .when(next.mask[0])
            .assert_eq(next.tidx, local.tidx + skip);
    }
}

pub(crate) fn generate_trace(proof: &Proof) -> RowMajorMatrix<F> {
    let num_valid_rows: usize = 1;
    let num_rows = num_valid_rows.next_power_of_two();
    let width = TranscriptCols::<usize>::width();

    let mut trace = vec![F::ZERO; num_rows * width];

    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut TranscriptCols<F> = row.borrow_mut();

        if i == 0 {
            // send common main stacking commit
            cols.tidx = F::ZERO;
            cols.mask = [F::ONE; CHUNK];
            // todo: initialize cols.preimage to hash(vkey.hash || [0; 8])
            // before overwriting.
            for j in 0..DIGEST_SIZE {
                cols.preimage[j] = proof.common_main_commit[j];
            }
        }
    }

    RowMajorMatrix::new(trace, width)
}
