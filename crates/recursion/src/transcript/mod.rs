use core::borrow::BorrowMut;
use std::sync::Arc;

use openvm_poseidon2_air::{POSEIDON2_WIDTH, Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{AirRef, p3_maybe_rayon::prelude::*, prover::MatrixDimensions};
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

// TODO: I think 1 is enough for now for our max constraint degree?
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
    #[tracing::instrument(name = "build_trace_artifacts(TranscriptModule)", skip_all)]
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
                cols.lookup[0] = F::from_canonical_usize(1);

                cols.prev_state = prev_poseidon_state;

                if is_sample {
                    cols.prev_state[CHUNK - 1] = preflight.transcript.values()[tidx];
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
                    cols.lookup[idx] = F::from_canonical_usize(1);
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
            if last_key.map_or(false, |prev| prev == key) {
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

    #[tracing::instrument(name = "generate_proving_ctxs(TranscriptModule)", skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKeyV2,
        proofs: &[Proof],
        preflights: &[Preflight],
        _ctx: &(),
    ) -> Vec<AirProvingContextV2<CpuBackendV2>> {
        let (merkle_verify_trace_vec, poseidon_inputs) =
            merkle_verify::generate_trace(child_vk, proofs, preflights, self.params.k_whir);
        let merkle_verify_trace =
            RowMajorMatrix::new(merkle_verify_trace_vec, MerkleVerifyCols::<F>::width());
        let TranscriptTraceArtifacts {
            transcript_trace,
            poseidon_inputs,
        } = self.build_trace_artifacts(preflights, poseidon_inputs);

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

        // Finally, make the RawInput structs
        let poseidon2_trace = RowMajorMatrix::new(poseidon_trace, poseidon2_width);
        [transcript_trace, poseidon2_trace, merkle_verify_trace]
            .map(|trace| AirProvingContextV2::simple_no_pis(ColMajorMatrix::from_row_major(&trace)))
            .into_iter()
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod cuda_tracegen {
    use cuda_backend_v2::{GpuBackendV2, transport_matrix_h2d_col_major};
    use itertools::Itertools;
    use openvm_cuda_backend::{base::DeviceMatrix, types::F as CudaF};
    use openvm_cuda_common::{
        copy::{MemCopyD2H, MemCopyH2D},
        d_buffer::DeviceBuffer,
    };

    use super::*;
    use crate::{
        cuda::{GlobalCtxGpu, preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
        transcript::cuda_abi,
    };

    impl TraceGenModule<GlobalCtxGpu, GpuBackendV2> for TranscriptModule {
        type ModuleSpecificCtx = ();

        #[tracing::instrument(name = "generate_proving_ctxs(TranscriptModule)", skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            _ctx: &(),
        ) -> Vec<AirProvingContextV2<GpuBackendV2>> {
            let proofs_cpu = proofs.iter().map(|proof| proof.cpu.clone()).collect_vec();
            let preflights_cpu = preflights
                .iter()
                .map(|preflight| preflight.cpu.clone())
                .collect_vec();
            let (merkle_trace, poseidon_inputs) = merkle_verify::generate_trace(
                &child_vk.cpu,
                &proofs_cpu,
                &preflights_cpu,
                self.params.k_whir,
            );
            let TranscriptTraceArtifacts {
                transcript_trace,
                poseidon_inputs,
            } = self.build_trace_artifacts(&preflights_cpu, poseidon_inputs);

            let transcript_trace_gpu =
                transport_matrix_h2d_col_major(&ColMajorMatrix::from_row_major(&transcript_trace))
                    .unwrap();
            let merkle_trace_gpu = transport_matrix_h2d_col_major(&ColMajorMatrix::from_row_major(
                &RowMajorMatrix::new(merkle_trace, MerkleVerifyCols::<F>::width()),
            ))
            .unwrap();

            let poseidon2_width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
            let poseidon2_valid_rows = poseidon_inputs.len();
            let poseidon_inputs_flat: Vec<CudaF> = poseidon_inputs
                .into_iter()
                .flat_map(|state| state.into_iter())
                .collect();
            let d_records = poseidon_inputs_flat.to_device().unwrap();
            let d_counts = if poseidon2_valid_rows == 0 {
                DeviceBuffer::<u32>::new()
            } else {
                vec![1u32; poseidon2_valid_rows].to_device().unwrap()
            };
            let mut num_records = poseidon2_valid_rows;
            if num_records > 0 {
                unsafe {
                    let d_num_records = [num_records].to_device().unwrap();
                    let mut temp_bytes = 0;
                    cuda_abi::poseidon2_deduplicate_records_get_temp_bytes(
                        &d_records,
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
                        &d_records,
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
                DeviceMatrix::<CudaF>::with_capacity(poseidon2_num_rows, poseidon2_width);
            unsafe {
                cuda_abi::poseidon2_tracegen(
                    poseidon_trace_gpu.buffer(),
                    poseidon_trace_gpu.height(),
                    poseidon_trace_gpu.width(),
                    &d_records,
                    &d_counts,
                    num_records,
                    SBOX_REGISTERS,
                )
                .unwrap();
            }

            vec![
                AirProvingContextV2::simple_no_pis(transcript_trace_gpu),
                AirProvingContextV2::simple_no_pis(poseidon_trace_gpu),
                AirProvingContextV2::simple_no_pis(merkle_trace_gpu),
            ]
        }
    }
}
