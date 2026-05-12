use std::array::from_fn;

// Re-export `DeferralFn` so `use openvm_deferral_circuit::DeferralFn` keeps
// working after the type moved upstream into `openvm-circuit`.
pub use openvm_circuit::arch::deferral::DeferralFn;
use openvm_circuit::arch::{
    deferral::{DeferralResult, DeferralState, InputCommit, InputMapVal, OutputCommit, OutputRaw},
    VmField,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{poseidon2::DeferralPoseidon2Chip, utils::f_commit_to_bytes};

#[derive(Clone, Debug, derive_new::new)]
pub struct RawDeferralResult {
    pub input: InputCommit,
    pub output_raw: OutputRaw,
}

/// Execute `def_fn` against `state`, hashing the output via `hasher` and
/// caching the result. Used to be `DeferralFn::execute`; moved here as a free
/// function because the inherent impl must live in `DeferralFn`'s home crate
/// (`openvm-circuit`), which can't depend on the Poseidon2 chip.
pub fn execute_deferral_fn<F: VmField>(
    def_fn: &DeferralFn,
    input_commit: &InputCommit,
    state: &mut DeferralState,
    deferral_idx: u32,
    hasher: &DeferralPoseidon2Chip<F>,
) -> (OutputCommit, u64) {
    match state.get_input(input_commit).clone() {
        InputMapVal::Raw(input_raw) => {
            let output_raw = def_fn.call_raw(&input_raw);
            let output_commit = hash_output_raw(hasher, deferral_idx, &output_raw);
            let output_len = output_raw.len();
            state.store_output(input_commit, output_commit.clone(), output_raw);
            (output_commit, output_len as u64)
        }
        InputMapVal::Output(output_commit) => {
            let output_len = state.get_output(&output_commit).len() as u64;
            (output_commit, output_len)
        }
    }
}

pub fn generate_deferral_results<F: VmField>(
    raw_results: Vec<RawDeferralResult>,
    deferral_idx: u32,
    hasher: &DeferralPoseidon2Chip<F>,
) -> Vec<DeferralResult> {
    raw_results
        .into_iter()
        .map(|r| {
            let output_commit = hash_output_raw(hasher, deferral_idx, &r.output_raw);
            DeferralResult {
                input: r.input,
                output_commit,
                output_raw: r.output_raw,
            }
        })
        .collect()
}

pub(crate) fn hash_output_raw<F: VmField>(
    hasher: &DeferralPoseidon2Chip<F>,
    deferral_idx: u32,
    output_ref: &[u8],
) -> OutputCommit {
    assert!(output_ref.len().is_multiple_of(DIGEST_SIZE));

    let mut state = [F::ZERO; POSEIDON2_WIDTH];
    state[0] = F::from_u32(deferral_idx);
    state[1] = F::from_usize(output_ref.len());

    let (lhs, rhs) = state_to_chunks(&state);
    if output_ref.is_empty() {
        let res = hasher.perm(&lhs, &rhs, true);
        return f_commit_to_bytes(&res).to_vec();
    }

    state[DIGEST_SIZE..].copy_from_slice(&hasher.perm(&lhs, &rhs, false));

    let mut output_chunks = output_ref.chunks_exact(DIGEST_SIZE);
    let last_chunk = output_chunks.next_back().unwrap();

    for chunk in output_chunks {
        let f_chunk = chunk.iter().map(|b| F::from_u8(*b)).collect::<Vec<_>>();
        state[..DIGEST_SIZE].copy_from_slice(&f_chunk);
        let (lhs, rhs) = state_to_chunks(&state);
        let capacity = hasher.perm(&lhs, &rhs, false);
        state[DIGEST_SIZE..].copy_from_slice(&capacity);
    }

    let (_, rhs) = state_to_chunks(&state);
    let last_chunk_f = from_fn(|i| F::from_u8(last_chunk[i]));
    let res = hasher.perm(&last_chunk_f, &rhs, true);
    f_commit_to_bytes(&res).to_vec()
}

pub(crate) fn chunks_to_state<F: Copy>(
    lhs: &[F; DIGEST_SIZE],
    rhs: &[F; DIGEST_SIZE],
) -> [F; POSEIDON2_WIDTH] {
    from_fn(|i| {
        if i < DIGEST_SIZE {
            lhs[i]
        } else {
            rhs[i - DIGEST_SIZE]
        }
    })
}

pub(crate) fn state_to_chunks<F: Copy>(
    state: &[F; POSEIDON2_WIDTH],
) -> ([F; DIGEST_SIZE], [F; DIGEST_SIZE]) {
    (from_fn(|i| state[i]), from_fn(|i| state[i + DIGEST_SIZE]))
}
