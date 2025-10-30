use core::borrow::BorrowMut;
use std::sync::Arc;

use openvm_poseidon2_air::{POSEIDON2_WIDTH, Poseidon2Config, Poseidon2SubChip};
use openvm_stark_backend::{AirRef, p3_maybe_rayon::prelude::*, prover::types::AirProofRawInput};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
use p3_air::BaseAir;
use p3_baby_bear::Poseidon2BabyBear;
use p3_field::FieldAlgebra;
use p3_matrix::dense::RowMajorMatrix;
use p3_symmetric::Permutation;
use stark_backend_v2::{
    F,
    keygen::types::MultiStarkVerifyingKeyV2,
    poseidon2::{
        poseidon2_perm,
        sponge::{FiatShamirTranscript, TranscriptHistory},
    },
    proof::Proof,
};

use crate::{
    system::{AirModule, BusInventory, Preflight},
    transcript::{
        merkle_verify::{MerkleVerifyAir, MerkleVerifyCols},
        poseidon2::{CHUNK, Poseidon2Air, Poseidon2Cols},
        transcript::{TranscriptAir, TranscriptCols},
    },
};

pub mod merkle_verify;
pub mod poseidon2;
pub mod transcript;

// TODO: I think 1 is enough for now for our max constraint degree?
const SBOX_REGISTERS: usize = 1;

pub struct TranscriptModule {
    mvk: Arc<MultiStarkVerifyingKeyV2>,
    pub bus_inventory: BusInventory,

    sub_chip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    perm: Poseidon2BabyBear<POSEIDON2_WIDTH>,
}

impl TranscriptModule {
    pub fn new(mvk: Arc<MultiStarkVerifyingKeyV2>, bus_inventory: BusInventory) -> Self {
        let sub_chip = Poseidon2SubChip::<F, 1>::new(Poseidon2Config::default().constants);
        Self {
            mvk,
            bus_inventory,
            sub_chip,
            perm: poseidon2_perm().clone(),
        }
    }
}

impl<TS: FiatShamirTranscript + TranscriptHistory> AirModule<TS> for TranscriptModule {
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

    fn run_preflight(&self, _proof: &Proof, _preflight: &mut Preflight, _ts: &mut TS) {}

    fn generate_proof_inputs(
        &self,
        proofs: &[Proof],
        preflights: &[Preflight],
    ) -> Vec<AirProofRawInput<F>> {
        // TODO: need to "extract" the poseidon2 lookups from merkle verify
        let merkle_verify_trace = merkle_verify::generate_trace(proofs, preflights);

        let transcript_width = TranscriptCols::<F>::width();
        let inner_width = self.sub_chip.air.width();
        let poseidon2_width = Poseidon2Cols::<F, SBOX_REGISTERS>::width();
        let mut valid_rows = Vec::with_capacity(preflights.len());

        let mut transcript_valid_rows = 0;
        // First pass, calculate number of rows for transcript
        for preflight in preflights.iter() {
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
        // TODO: need to also add poseidon2 lookups from merkle verify
        let mut poseidon_inputs = Vec::with_capacity(transcript_valid_rows);
        let mut poseidon_flags = Vec::with_capacity(transcript_valid_rows);

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
                        cols.prev_state[CHUNK - 1 - idx] = preflight.transcript.values()[tidx];
                    } else {
                        cols.prev_state[idx] = preflight.transcript.values()[tidx];
                    }

                    tidx += 1;
                    idx += 1;
                    if idx == CHUNK {
                        permuted = true;
                        break;
                    }
                }
                cols.permuted = F::from_bool(permuted);

                prev_poseidon_state = cols.prev_state;
                if permuted {
                    self.perm.permute_mut(&mut prev_poseidon_state);
                    poseidon_inputs.push(cols.prev_state);
                    poseidon_flags.push(F::ONE);
                }
                cols.post_state = prev_poseidon_state;
            }
            skip += valid_rows[pidx];
            assert_eq!(tidx, preflight.transcript.len());
        }

        // TODO: add poseidon2 lookups from merkle verify
        poseidon_inputs.resize(transcript_num_rows, [F::ZERO; POSEIDON2_WIDTH]);
        poseidon_flags.resize(transcript_num_rows, F::ZERO);
        let inner_trace = self.sub_chip.generate_trace(poseidon_inputs);

        let mut poseidon_trace = F::zero_vec(transcript_num_rows * poseidon2_width);
        poseidon_trace
            .par_chunks_mut(poseidon2_width)
            .zip(inner_trace.values.par_chunks(inner_width))
            .zip(poseidon_flags)
            .for_each(|((row, inner_row), mult)| {
                row[..inner_width].copy_from_slice(inner_row);
                let cols: &mut Poseidon2Cols<F, SBOX_REGISTERS> = row.borrow_mut();
                cols.mult = mult;
            });

        // Finally, make the RawInput structs
        let transcript_input = AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                transcript_trace,
                transcript_width,
            ))),
            public_values: vec![],
        };
        let poseidon2_input = AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                poseidon_trace,
                poseidon2_width,
            ))),
            public_values: vec![],
        };
        let merkle_verify_input = AirProofRawInput {
            cached_mains: vec![],
            common_main: Some(Arc::new(RowMajorMatrix::new(
                merkle_verify_trace,
                MerkleVerifyCols::<F>::width(),
            ))),
            public_values: vec![],
        };
        vec![transcript_input, poseidon2_input, merkle_verify_input]
    }
}
