use core::borrow::BorrowMut;
use std::sync::Arc;

use openvm_poseidon2_air::{POSEIDON2_WIDTH, Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{AirRef, p3_maybe_rayon::prelude::*};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_air::BaseAir;
use p3_baby_bear::Poseidon2BabyBear;
use p3_field::{FieldAlgebra, PrimeField32};
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Permutation;
use stark_backend_v2::{
    F, SystemParams,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::poseidon2_perm,
    proof::Proof,
    prover::{AirProvingContextV2, ColMajorMatrix, CpuBackendV2},
};
use tracing::trace_span;

use crate::{
    system::{AirModule, BusInventory, GlobalCtxCpu, Preflight, TraceGenModule},
    transcript::{
        merkle_verify::{MerkleVerifyAir, MerkleVerifyCols},
        poseidon2::{CHUNK, Poseidon2Air, Poseidon2Cols},
        transcript::{TranscriptAir, TranscriptCols},
    },
};

#[cfg(feature = "cuda")]
mod cuda_abi;
pub mod merkle_verify;
pub mod poseidon2;
#[allow(clippy::module_inception)]
pub mod transcript;

// Should be 1 when 3 <= max_constraint_degree < 7
const SBOX_REGISTERS: usize = 1;

pub struct TranscriptModule {
    pub bus_inventory: BusInventory,
    params: SystemParams,

    sub_chip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    perm: Poseidon2BabyBear<POSEIDON2_WIDTH>,
}

impl TranscriptModule {
    pub fn new(bus_inventory: BusInventory, params: SystemParams) -> Self {
        let sub_chip = Poseidon2SubChip::<F, 1>::new(Poseidon2Config::default().constants);
        Self {
            bus_inventory,
            params,
            sub_chip,
            perm: poseidon2_perm().clone(),
        }
    }

    // Builds trace for transcript and merkle verify AIRs (and records poseidon2 permutations).
    // Also combines in the poseidon2 permutations from preflight (from WHIR).
    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    fn build_trace_artifacts(
        &self,
        preflights: &[Preflight],
        mut poseidon2_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> TranscriptTraceArtifacts {
        let transcript_width = TranscriptCols::<F>::width();
        let mut valid_rows = Vec::with_capacity(preflights.len());

        let mut transcript_valid_rows = 0;
        // First pass, calculate number of rows for transcript
        for preflight in preflights.iter() {
            poseidon2_inputs.extend_from_slice(&preflight.poseidon_inputs);
            let mut cur_is_sample = false; // should always start with observe?
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
            valid_rows.push(num_valid_rows);
            transcript_valid_rows += num_valid_rows;
        }
        let transcript_num_rows = transcript_valid_rows.next_power_of_two();
        let mut transcript_trace = vec![F::ZERO; transcript_num_rows * transcript_width];

        let mut skip = 0;
        // Second pass, fill in the transcript trace.
        for (pidx, preflight) in preflights.iter().enumerate() {
            let mut tidx = 0;
            let mut prev_poseidon_state = [F::ZERO; POSEIDON2_WIDTH];
            let off = skip * transcript_width;
            let end = off + valid_rows[pidx] * transcript_width;
            for (i, row) in transcript_trace[off..end]
                .chunks_exact_mut(transcript_width)
                .enumerate()
            {
                let cols: &mut TranscriptCols<F> = row.borrow_mut();
                cols.proof_idx = F::from_canonical_usize(pidx);
                if i == 0 {
                    cols.is_proof_start = F::ONE;
                }
                let is_sample = preflight.transcript.samples()[tidx];

                cols.is_sample = F::from_bool(is_sample);
                cols.tidx = F::from_canonical_usize(tidx);
                cols.mask[0] = F::from_bool(true);

                cols.prev_state = prev_poseidon_state;

                if is_sample {
                    debug_assert_eq!(
                        cols.prev_state[CHUNK - 1],
                        preflight.transcript.values()[tidx],
                        "sample value mismatch",
                    );
                } else {
                    cols.prev_state[0] = preflight.transcript.values()[tidx];
                }

                tidx += 1;
                let mut idx: usize = 1;

                let mut permuted = false;
                loop {
                    if tidx >= preflight.transcript.len() {
                        // at the end, no permutation needed
                        break;
                    }

                    if preflight.transcript.samples()[tidx] != is_sample {
                        // encounter a different type of operation. Permute if it's going to sample
                        permuted = preflight.transcript.samples()[tidx];
                        break;
                    }

                    cols.mask[idx] = F::from_bool(true);
                    if is_sample {
                        debug_assert_eq!(
                            cols.prev_state[CHUNK - 1 - idx],
                            preflight.transcript.values()[tidx],
                            "sample value mismatch",
                        );
                    } else {
                        cols.prev_state[idx] = preflight.transcript.values()[tidx];
                    }

                    tidx += 1;
                    idx += 1;
                    if idx == CHUNK {
                        // If it's sample -> observe, we don't need to permute. otherwise permute
                        permuted = !is_sample || preflight.transcript.samples()[tidx];
                        break;
                    }
                }
                cols.permuted = F::from_bool(permuted);

                prev_poseidon_state = cols.prev_state;
                if permuted {
                    self.perm.permute_mut(&mut prev_poseidon_state);
                    poseidon2_inputs.push(cols.prev_state);
                }
                cols.post_state = prev_poseidon_state;
            }
            skip += valid_rows[pidx];
            assert_eq!(tidx, preflight.transcript.len());
        }

        TranscriptTraceArtifacts {
            transcript_trace: RowMajorMatrix::new(transcript_trace, transcript_width),
            poseidon_inputs: poseidon2_inputs,
        }
    }

    fn dedup_poseidon_inputs(
        poseidon_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<u32>) {
        let mut keyed_states = poseidon_inputs
            .into_iter()
            .map(|state| (state.map(|x| x.as_canonical_u32()), state))
            .collect::<Vec<_>>();
        keyed_states.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let mut deduped = Vec::with_capacity(keyed_states.len());
        let mut counts = Vec::new();
        let mut last_key: Option<[u32; POSEIDON2_WIDTH]> = None;
        for (key, state) in keyed_states {
            if last_key == Some(key) {
                if let Some(last) = counts.last_mut() {
                    *last += 1;
                }
            } else {
                deduped.push(state);
                counts.push(1);
                last_key = Some(key);
            }
        }
        (deduped, counts)
    }
}

impl AirModule for TranscriptModule {
    fn airs(&self) -> Vec<AirRef<BabyBearPoseidon2Config>> {
        let transcript_air = TranscriptAir {
            transcript_bus: self.bus_inventory.transcript_bus,
            poseidon2_bus: self.bus_inventory.poseidon2_bus,
        };
        let poseidon2_air = Poseidon2Air::<F, SBOX_REGISTERS> {
            poseidon2_bus: self.bus_inventory.poseidon2_bus,
            subair: self.sub_chip.air.clone(),
        };
        let merkle_verify_air = MerkleVerifyAir {
            poseidon2_bus: self.bus_inventory.poseidon2_bus,
            merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
            commitments_bus: self.bus_inventory.commitments_bus,
        };
        vec![
            Arc::new(transcript_air),
            Arc::new(poseidon2_air),
            Arc::new(merkle_verify_air),
        ]
    }
}

pub(super) struct TranscriptTraceArtifacts {
    transcript_trace: RowMajorMatrix<F>,
    poseidon_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl TraceGenModule<GlobalCtxCpu, CpuBackendV2> for TranscriptModule {
    type ModuleSpecificCtx = ();

    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        _ctx: &(),
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let (merkle_verify_trace_vec, poseidon_inputs) =
            tracing::info_span!("wrapper.generate_trace", air = "MerkleVerify").in_scope(|| {
                merkle_verify::generate_trace(child_vk, proofs, preflights, &self.params)
            });
        let merkle_verify_trace =
            RowMajorMatrix::new(merkle_verify_trace_vec, MerkleVerifyCols::<F>::width());
        let TranscriptTraceArtifacts {
            transcript_trace,
            poseidon_inputs,
        } = tracing::trace_span!("wrapper.generate_trace", air = "Transcript")
            .in_scope(|| self.build_trace_artifacts(preflights, poseidon_inputs));

        let poseidon2_trace =
            trace_span!("wrapper.generate_trace", air = "Poseidon2").in_scope(|| {
                // TODO: This is unfortunately how we propagate span fields given our current
                // tracing system. It would be extraordinarily helpful to update our
                // metric outputs to contain the fields they define as labels.
                trace_span!("generate_trace").in_scope(|| {
                    let (mut poseidon_states, mut poseidon_counts) =
                        Self::dedup_poseidon_inputs(poseidon_inputs);
                    let poseidon2_valid_rows = poseidon_states.len();
                    let poseidon2_num_rows = if poseidon2_valid_rows == 0 {
                        1
                    } else {
                        poseidon2_valid_rows.next_power_of_two()
                    };
                    poseidon_states.resize(poseidon2_num_rows, [F::ZERO; POSEIDON2_WIDTH]);
                    poseidon_counts.resize(poseidon2_num_rows, 0);
                    let poseidon_counts: Vec<F> = poseidon_counts
                        .into_iter()
                        .map(F::from_canonical_u32)
                        .collect();

                    let inner_width = self.sub_chip.air.width();
                    let poseidon2_width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
                    let inner_trace = self.sub_chip.generate_trace(poseidon_states);
                    let mut poseidon_trace = F::zero_vec(poseidon2_num_rows * poseidon2_width);
                    poseidon_trace
                        .par_chunks_mut(poseidon2_width)
                        .zip(inner_trace.values.par_chunks(inner_width))
                        .enumerate()
                        .for_each(|(i, (row, inner_row))| {
                            row[..inner_width].copy_from_slice(inner_row);
                            let cols: &mut Poseidon2Cols<F, SBOX_REGISTERS> = row.borrow_mut();
                            cols.mult = poseidon_counts.get(i).copied().unwrap_or(F::ZERO);
                        });
                    RowMajorMatrix::new(poseidon_trace, poseidon2_width)
                })
            });

        // Finally, make the RawInput structs
        [transcript_trace, poseidon2_trace, merkle_verify_trace]
            .map(|trace| AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)))
            .into_iter()
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use cuda_backend_v2::GpuBackendV2;
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, types::F};
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
    };
    use openvm_stark_backend::prover::MatrixDimensions;

    use super::*;
    use crate::{
        cuda::{GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
        transcript::{
            cuda_abi,
            merkle_verify::{self, cuda::MerkleVerifyBlob},
            transcript::cuda::TranscriptAirBlob,
        },
    };

    pub(crate) struct TranscriptBlob {
        pub merkle_verify_blob: MerkleVerifyBlob,
        pub transcript_air_blob: TranscriptAirBlob,
        pub poseidon2_buffer: DeviceBuffer<F>,
        pub num_poseidon2_inputs: usize,
    }

    impl TranscriptBlob {
        #[tracing::instrument(name = "generate_blob", skip_all)]
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
        ) -> Self {
            let poseidon2_inputs = preflights
                .iter()
                .flat_map(|preflight| preflight.cpu.poseidon_inputs.clone())
                .collect_vec();
            let mut num_poseidon2_inputs = poseidon2_inputs.len();

            let merkle_verify_blob =
                MerkleVerifyBlob::new(child_vk, proofs, preflights, num_poseidon2_inputs);
            num_poseidon2_inputs += merkle_verify_blob.total_rows;

            let transcript_air_blob =
                TranscriptAirBlob::new(preflights, num_poseidon2_inputs as u32);
            num_poseidon2_inputs += transcript_air_blob.num_poseidon2_inputs;

            let mut poseidon2_buffer =
                DeviceBuffer::with_capacity(num_poseidon2_inputs * POSEIDON2_WIDTH);
            poseidon2_inputs
                .into_iter()
                .flatten()
                .collect_vec()
                .copy_to(&mut poseidon2_buffer)
                .unwrap();

            Self {
                merkle_verify_blob,
                transcript_air_blob,
                poseidon2_buffer,
                num_poseidon2_inputs,
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for TranscriptModule {
        type ModuleSpecificCtx = ();

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _ctx: &(),
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            let blob = TranscriptBlob::new(child_vk, proofs, preflights);

            let merkle_trace = tracing::trace_span!("wrapper.generate_trace", air = "MerkleVerify")
                .in_scope(|| merkle_verify::cuda::generate_trace(&blob));
            let transcript_trace =
                tracing::trace_span!("wrapper.generate_trace", air = "Transcript")
                    .in_scope(|| transcript::cuda::generate_trace(preflights, &blob));
            let poseidon_trace =
                trace_span!("wrapper.generate_trace", air = "Poseidon2").in_scope(|| {
                    trace_span!("generate_trace").in_scope(|| {
                        let poseidon2_width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();

                        let d_counts = if blob.num_poseidon2_inputs == 0 {
                            DeviceBuffer::<u32>::new()
                        } else {
                            vec![1u32; blob.num_poseidon2_inputs].to_device().unwrap()
                        };

                        let mut num_records = blob.num_poseidon2_inputs;
                        if num_records > 0 {
                            unsafe {
                                let d_num_records = [num_records].to_device().unwrap();
                                let mut temp_bytes = 0;
                                cuda_abi::poseidon2_deduplicate_records_get_temp_bytes(
                                    &blob.poseidon2_buffer,
                                    &d_counts,
                                    num_records,
                                    &d_num_records,
                                    &mut temp_bytes,
                                )
                                .unwrap();
                                let d_temp_storage = if temp_bytes == 0 {
                                    DeviceBuffer::<u8>::new()
                                } else {
                                    DeviceBuffer::<u8>::with_capacity(temp_bytes)
                                };
                                cuda_abi::poseidon2_deduplicate_records(
                                    &blob.poseidon2_buffer,
                                    &d_counts,
                                    num_records,
                                    &d_num_records,
                                    &d_temp_storage,
                                    temp_bytes,
                                )
                                .unwrap();
                                num_records = *d_num_records.to_host().unwrap().first().unwrap();
                            }
                        }
                        let poseidon2_num_rows = if num_records == 0 {
                            1
                        } else {
                            num_records.next_power_of_two()
                        };
                        let poseidon_trace_gpu =
                            DeviceMatrix::<F>::with_capacity(poseidon2_num_rows, poseidon2_width);
                        unsafe {
                            cuda_abi::poseidon2_tracegen(
                                poseidon_trace_gpu.buffer(),
                                poseidon_trace_gpu.height(),
                                poseidon_trace_gpu.width(),
                                &blob.poseidon2_buffer,
                                &d_counts,
                                num_records,
                                SBOX_REGISTERS,
                            )
                            .unwrap();
                        }
                        poseidon_trace_gpu
                    })
                });

            vec![
                AirProvingContextV2::simple_no_pis(transcript_trace),
                AirProvingContextV2::simple_no_pis(poseidon_trace),
                AirProvingContextV2::simple_no_pis(merkle_trace),
            ]
        }
    }
}
