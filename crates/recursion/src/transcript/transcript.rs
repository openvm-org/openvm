use core::borrow::Borrow;

use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;
use stark_recursion_circuit_derive::AlignedBorrow;

use crate::{
    bus::{Poseidon2PermuteBus, Poseidon2PermuteMessage, TranscriptBus, TranscriptBusMessage},
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
    /// the transcript). Constrained to be "decreasing". Because transcript_bus messages
    /// are sent with multiplicity 0 or 1, this also functions as the lookup count.
    pub mask: [T; CHUNK],

    /// indicator, whether the state is permutation from previous row's state
    pub permuted: T,
    /// The poseidon2 state.
    pub prev_state: [T; POSEIDON2_WIDTH],
    pub post_state: [T; POSEIDON2_WIDTH],
}

pub struct TranscriptAir {
    pub transcript_bus: TranscriptBus,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
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
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let local: &TranscriptCols<AB::Var> = (*local).borrow();
        let next: &TranscriptCols<AB::Var> = (*next).borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////
        let is_valid = local.mask[0];

        builder.when(local.is_proof_start).assert_zero(local.tidx);
        builder.when(local.is_proof_start).assert_one(is_valid);
        builder.assert_bool(local.is_sample);

        // Initial state constraints
        for i in 0..CHUNK {
            builder
                .when(local.is_proof_start)
                .assert_eq(local.prev_state[i + CHUNK], AB::Expr::ZERO);

            builder
                .when(local.is_proof_start * (AB::Expr::ONE - local.mask[i]))
                .assert_eq(local.prev_state[i], AB::Expr::ZERO);
        }

        let mut count = AB::Expr::ZERO;
        let local_next_same_proof = next.mask[0] * (AB::Expr::ONE - next.is_proof_start);
        for i in 0..CHUNK {
            builder.assert_bool(local.mask[i]);
            count += local.mask[i].into();

            let skip = local.mask[i] - AB::Expr::ONE;
            if i < CHUNK - 1 {
                // if mask[i] = 0, then mask[i+1] = 0
                builder.when(skip.clone()).assert_zero(local.mask[i + 1]);
            }

            // The state after permutation of this round, should check against next round's input
            // if next.mask[i] = 0 --> i-th not touched --> it should stay the same (if next is
            // valid)
            builder
                .when((AB::Expr::ONE - next.mask[i]) * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);
            // When it's squeeze(sample), the state always remains the same
            builder
                .when(next.is_sample * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);

            // The capacity part should always be the same
            builder
                .when(local_next_same_proof.clone()) // if next is valid
                .assert_eq(local.post_state[i + CHUNK], next.prev_state[i + CHUNK]);
        }

        builder
            .when(local_next_same_proof) // if next is valid
            .assert_eq(next.tidx, local.tidx + count);

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////
        for i in 0..CHUNK {
            // When absorb, it's normal order (0 -> RATE)
            let observe_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(i),
                value: local.prev_state[i].into(),
                is_sample: AB::Expr::ZERO,
            };
            // When squeeze, it's reverse RATE -> 0, so i means RATE - 1 - i
            let sample_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(i),
                value: local.prev_state[CHUNK - 1 - i].into(),
                is_sample: AB::Expr::ONE,
            };
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                observe_message,
                local.mask[i] * (AB::Expr::ONE - local.is_sample),
            );
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                sample_message,
                local.mask[i] * local.is_sample,
            );
        }

        self.poseidon2_permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.prev_state,
                output: local.post_state,
            },
            local.permuted,
        )
    }
}

#[cfg(feature = "cuda")]
pub mod cuda {
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, prelude::F};
    use openvm_cuda_common::copy::MemCopyH2D;

    use crate::{
        cuda::preflight::PreflightGpu,
        transcript::{
            cuda_abi::transcript_air_tracegen, cuda_tracegen::TranscriptBlob, poseidon2::CHUNK,
            transcript::TranscriptCols,
        },
    };

    #[repr(C)]
    pub(crate) struct TranscriptAirRecord {
        pub is_sample: bool,
        pub permuted: bool,
        pub num_ops: u8,
        pub tidx: u32,
        pub state_idx: u32,
    }

    pub(crate) struct TranscriptAirBlob {
        pub records: Vec<Vec<TranscriptAirRecord>>,
        pub poseidon2_offsets: Vec<u32>,
        pub num_poseidon2_perms: usize,
    }

    impl TranscriptAirBlob {
        pub fn new(preflights: &[PreflightGpu], starting_poseidon_idx: u32) -> Self {
            let mut records = vec![];
            let mut poseidon2_offsets = vec![starting_poseidon_idx];
            for preflight in preflights {
                let mut local_records = vec![];
                let samples = preflight.cpu.transcript.samples();

                let mut is_sample = samples[0];
                let mut first_idx = 0;
                let mut num_ops = 1u8;
                let mut state_idx = 0u32;

                for (tidx, &next) in samples.iter().enumerate().skip(1) {
                    if next == is_sample {
                        if num_ops as usize == CHUNK {
                            local_records.push(TranscriptAirRecord {
                                is_sample,
                                permuted: true,
                                num_ops,
                                tidx: first_idx,
                                state_idx,
                            });
                            first_idx = tidx as u32;
                            num_ops = 1;
                            state_idx += 1;
                        } else {
                            num_ops += 1;
                        }
                    } else {
                        local_records.push(TranscriptAirRecord {
                            is_sample,
                            permuted: !is_sample,
                            num_ops,
                            tidx: first_idx,
                            state_idx,
                        });
                        first_idx = tidx as u32;
                        num_ops = 1;
                        state_idx += (!is_sample) as u32;
                    }
                    is_sample = next;
                }

                // Emit the final (possibly partial) segment. By construction the transcript ends
                // after this row, so no permutation is required for continuity.
                if !samples.is_empty() {
                    local_records.push(TranscriptAirRecord {
                        is_sample,
                        permuted: false,
                        num_ops,
                        tidx: first_idx,
                        state_idx,
                    });
                }

                records.push(local_records);
                poseidon2_offsets.push(poseidon2_offsets.last().unwrap() + state_idx);
            }

            let num_poseidon2_inputs =
                (poseidon2_offsets.last().unwrap() - poseidon2_offsets.first().unwrap()) as usize;
            poseidon2_offsets.pop();

            Self {
                records,
                poseidon2_offsets,
                num_poseidon2_perms: num_poseidon2_inputs,
            }
        }
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub(crate) fn generate_trace(
        preflights_gpu: &[PreflightGpu],
        blob: &TranscriptBlob,
        required_height: Option<usize>,
    ) -> Option<DeviceMatrix<F>> {
        let mut num_valid_rows = 0usize;
        let mut row_bounds = Vec::with_capacity(preflights_gpu.len());

        let records = blob
            .transcript_air_blob
            .records
            .iter()
            .map(|v| {
                num_valid_rows += v.len();
                row_bounds.push(num_valid_rows as u32);
                v.to_device().unwrap()
            })
            .collect_vec();
        let transcript_values = preflights_gpu
            .iter()
            .map(|preflight| preflight.cpu.transcript.values().to_device().unwrap())
            .collect_vec();
        let start_states = preflights_gpu
            .iter()
            .map(|preflight| {
                preflight
                    .cpu
                    .transcript
                    .perm_results()
                    .iter()
                    .flatten()
                    .copied()
                    .collect_vec()
                    .to_device()
                    .unwrap()
            })
            .collect_vec();

        let height = if let Some(height) = required_height {
            if height < num_valid_rows {
                return None;
            }
            height
        } else {
            num_valid_rows.next_power_of_two()
        };
        let width = TranscriptCols::<usize>::width();
        let d_trace = DeviceMatrix::with_capacity(height, width);

        let d_transcript_values = transcript_values.iter().map(|b| b.as_ptr()).collect_vec();
        let d_start_states = start_states.iter().map(|b| b.as_ptr()).collect_vec();
        let d_records = records.iter().map(|b| b.as_ptr()).collect_vec();

        unsafe {
            transcript_air_tracegen(
                d_trace.buffer(),
                height,
                width,
                &row_bounds,
                d_transcript_values,
                d_start_states,
                d_records,
                &blob.poseidon2_buffer,
                &blob.transcript_air_blob.poseidon2_offsets,
                preflights_gpu.len(),
            )
            .unwrap();
        }

        Some(d_trace)
    }
}
