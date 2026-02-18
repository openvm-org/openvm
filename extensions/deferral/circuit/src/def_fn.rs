use std::array::from_fn;

use openvm_circuit::arch::{
    deferral::{
        DeferralResult, DeferralState, InputCommit, InputMapVal, InputRaw, OutputCommit, OutputRaw,
    },
    hasher::Hasher,
    VmField,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{poseidon2::DeferralPoseidon2Chip, utils::f_commit_to_bytes};

#[derive(Clone, Debug, derive_new::new)]
pub struct RawDeferralResult {
    pub input: InputCommit,
    pub output_raw: OutputRaw,
}

pub struct DeferralFn {
    f: Box<dyn Fn(&InputRaw) -> OutputRaw>,
}

impl DeferralFn {
    pub fn new<FN: Fn(&InputRaw) -> OutputRaw + 'static>(f: FN) -> Self {
        Self { f: Box::new(f) }
    }

    pub fn execute<F: VmField>(
        &self,
        input_commit: &InputCommit,
        state: &mut DeferralState,
        deferral_idx: u32,
        hasher: &DeferralPoseidon2Chip<F>,
    ) -> (OutputCommit, u32) {
        let value = state.get_input(input_commit);
        match value {
            InputMapVal::Raw(input_raw) => {
                let output_raw = self.f.as_ref()(input_raw);
                let output_commit = hash_output_raw(hasher, deferral_idx, &output_raw);
                let output_len = output_raw.len();
                state.store_output(input_commit, output_commit.clone(), output_raw);
                (output_commit, output_len as u32)
            }
            InputMapVal::Output(output_commit) => {
                let output_raw = state.get_output(output_commit);
                (output_commit.clone(), output_raw.len() as u32)
            }
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

fn hash_output_raw<F: VmField>(
    hasher: &DeferralPoseidon2Chip<F>,
    deferral_idx: u32,
    output_ref: &[u8],
) -> OutputCommit {
    assert_eq!(output_ref.len() % DIGEST_SIZE, 0);
    let mut state = [F::ZERO; DIGEST_SIZE];
    state[0] = F::from_u32(deferral_idx);
    for chunk in output_ref.chunks_exact(DIGEST_SIZE) {
        let bytes = from_fn(|i| F::from_u8(chunk[i]));
        state = hasher.compress(&state, &bytes);
    }
    f_commit_to_bytes(&state).to_vec()
}
