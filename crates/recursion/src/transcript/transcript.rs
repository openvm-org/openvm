use core::borrow::{Borrow, BorrowMut};

use crate::bus::{Poseidon2Bus, Poseidon2BusMessage, TranscriptBus};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, FieldAlgebra};
use p3_matrix::Matrix;
use stark_backend_v2::{F, proof::Proof};
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::TranscriptBusMessage,
    system::Preflight,
    transcript::poseidon2::{CHUNK, POSEIDON2_WIDTH},
};

#[repr(C)]
#[derive(AlignedBorrow, Debug)]
pub struct TranscriptCols<T> {
    pub proof_idx: T,
    pub is_proof_start: T,

    pub tidx: T,
    /// Indicator for sample/observe.
    pub is_sample: T,
    /// 0/1 indicators for the positions that we are absorbing/squeezing (i.e., are in
    /// the transcript). Constrained to be "decreasing".
    pub mask: [T; CHUNK],
    /// The lookup counts. Must be zero when corresponding mask is zero; otherwise unconstrained.
    pub lookup: [T; CHUNK],

    /// inidicator, whether the state is permutation from previous row's state
    pub permuted: T,
    /// The poseidon2 state.
    pub state: [T; POSEIDON2_WIDTH],
}

pub struct TranscriptAir {
    pub transcript_bus: TranscriptBus,
    pub poseidon2_bus: Poseidon2Bus,
}

impl<F: Field> BaseAir<F> for TranscriptAir {
    fn width(&self) -> usize {
        TranscriptCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TranscriptAir {}
impl<F: Field> PartitionedBaseAir<F> for TranscriptAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for TranscriptAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &TranscriptCols<AB::Var> = (*local).borrow();
        let next: &TranscriptCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////
        for i in 0..CHUNK {
            // When absorb, it's normal order (0 -> RATE)
            let observe_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                value: local.state[i].into(),
                is_sample: AB::Expr::ZERO,
            };
            // When squeeze, it's reverse RATE -> 0, so i means RATE - 1 - i
            let sample_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_canonical_usize(i),
                value: local.state[CHUNK - 1 - i].into(),
                is_sample: AB::Expr::ONE,
            };
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                observe_message,
                local.lookup[i] * (AB::Expr::ONE - local.is_sample),
            );
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                sample_message,
                local.lookup[i] * local.is_sample,
            );
        }

        self.poseidon2_bus.lookup_key(
            builder,
            Poseidon2BusMessage {
                input: local.state,
                output: next.state,
            },
            next.permuted,
        )
    }
}

pub fn generate_trace(_proof: &Proof, preflight: &Preflight) -> Vec<F> {
    let width = TranscriptCols::<F>::width();

    // First pass, just calculate number of rows.
    let mut cur_is_sample = false; // should start with observe
    let mut count = 0;
    let mut num_valid_rows: usize = 0;
    for op_is_sample in preflight.transcript.samples() {
        if *op_is_sample {
            // sample
            if !cur_is_sample {
                // observe -> sample, need a new row and permute
                num_valid_rows += 1;
                cur_is_sample = true;
                count = 1;
            } else {
                if count == CHUNK {
                    num_valid_rows += 1;
                    count = 0;
                }
                count += 1;
            }
        } else {
            // observe
            if cur_is_sample {
                // sample -> observe, no need to permute, but still need a new row
                num_valid_rows += 1;
                cur_is_sample = false;
                count = 1;
            } else {
                if count == CHUNK {
                    num_valid_rows += 1;
                    count = 0;
                }
                count += 1;
            }
        }
    }
    if count > 0 {
        num_valid_rows += 1;
    }

    let num_rows = num_valid_rows.next_power_of_two();
    let mut trace = vec![F::ZERO; num_rows.next_power_of_two() * width];

    // Second pass, fill in the trace.
    // TODO: the poseidon2 state is not correct yet.
    let mut tidx = 0;
    let mut prev_is_sample = false;
    for (i, row) in trace.chunks_mut(width).take(num_valid_rows).enumerate() {
        let cols: &mut TranscriptCols<F> = row.borrow_mut();
        if i == 0 {
            cols.is_proof_start = F::ONE;
        }
        let is_sample = preflight.transcript.samples()[tidx];
        cols.is_sample = F::from_bool(is_sample);
        cols.tidx = F::from_canonical_usize(tidx);
        cols.mask[0] = F::from_bool(true);
        cols.lookup[0] = F::from_canonical_usize(1);

        if is_sample {
            cols.state[CHUNK - 1] = preflight.transcript.values()[tidx];
        } else {
            cols.state[0] = preflight.transcript.values()[tidx];
        }

        tidx += 1;
        let mut idx: usize = 1;

        loop {
            if tidx >= preflight.transcript.len()
                || preflight.transcript.samples()[tidx] != is_sample
            {
                break;
            }

            cols.mask[idx] = F::from_bool(true);
            cols.lookup[idx] = F::from_canonical_usize(1);
            if is_sample {
                cols.state[CHUNK - 1 - idx] = preflight.transcript.values()[tidx];
            } else {
                cols.state[idx] = preflight.transcript.values()[tidx];
            }

            tidx += 1;
            idx += 1;
            if idx == CHUNK {
                break;
            }
        }

        if prev_is_sample && !is_sample || i == 0 {
            // previous row is sample, current row is observe --> no need to permute
            // Also no permutation for the first row
            cols.permuted = F::ZERO;
        } else {
            // in all other cases, we need to permute
            cols.permuted = F::ONE;
        }
        prev_is_sample = is_sample;
    }
    assert_eq!(tidx, preflight.transcript.len());

    trace
}
