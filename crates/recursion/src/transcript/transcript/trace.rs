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
