#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{vec, vec::Vec};

use openvm_deferral_guest::{deferred_compute, get_deferred_output, Commit, COMMIT_NUM_BYTES};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofOutput {
    pub app_exe_commit: Commit,
    pub app_vm_commit: Commit,
    pub num_public_values: u32,
    pub user_public_values: Vec<u8>,
}

pub fn verify_stark_unchecked<const DEF_IDX: u16>(input_commit: &Commit) -> ProofOutput {
    let output_key = deferred_compute::<DEF_IDX>(input_commit);
    let output_len = output_key.output_len as usize;
    let mut output_bytes = vec![0u8; output_len];
    get_deferred_output::<DEF_IDX>(&mut output_bytes, &output_key);

    const COMMITS_OUTPUT_BYTES: usize = 2 * COMMIT_NUM_BYTES;
    const PUBLIC_VALUES_HEADER_BYTES: usize = 2 * core::mem::size_of::<u32>();
    const MIN_OUTPUT_BYTES: usize = COMMITS_OUTPUT_BYTES + PUBLIC_VALUES_HEADER_BYTES;
    if output_len < MIN_OUTPUT_BYTES {
        panic!("output_len too small for a ProofOutput");
    }

    let app_exe_commit = output_bytes[..COMMIT_NUM_BYTES].try_into().unwrap();
    let app_vm_commit = output_bytes[COMMIT_NUM_BYTES..COMMITS_OUTPUT_BYTES]
        .try_into()
        .unwrap();
    let public_values_output = &output_bytes[COMMITS_OUTPUT_BYTES..];
    let (num_public_values, user_public_values) = decode_user_public_values(public_values_output);

    ProofOutput {
        app_exe_commit,
        app_vm_commit,
        num_public_values,
        user_public_values,
    }
}

pub fn verify_stark<const DEF_IDX: u16>(input_commit: &Commit, expected: &ProofOutput) {
    let actual = verify_stark_unchecked::<DEF_IDX>(input_commit);
    if actual != *expected {
        panic!("Proof verification failed for commit {:?}", input_commit);
    }
}

fn decode_user_public_values(expanded: &[u8]) -> (u32, Vec<u8>) {
    const F_NUM_BYTES: usize = core::mem::size_of::<u32>();
    const U16_CELL_SIZE: usize = core::mem::size_of::<u16>();
    const PUBLIC_VALUE_BYTES: usize = core::mem::size_of::<u64>();

    if expanded.len() < 2 * F_NUM_BYTES {
        panic!("User public values output is missing its header");
    }
    if !expanded.len().is_multiple_of(F_NUM_BYTES) {
        panic!("User public values output length is not a multiple of {F_NUM_BYTES}");
    }
    let num_public_values = u32::from_le_bytes(
        expanded[..F_NUM_BYTES]
            .try_into()
            .expect("count is one field element"),
    );
    if expanded[F_NUM_BYTES..2 * F_NUM_BYTES]
        .iter()
        .any(|&byte| byte != 0)
    {
        panic!("User public values reserved field is non-zero");
    }

    let cells = &expanded[2 * F_NUM_BYTES..];
    let mut user_public_values = Vec::with_capacity(cells.len() / F_NUM_BYTES * U16_CELL_SIZE);
    for bytes in cells.chunks_exact(F_NUM_BYTES) {
        if bytes[U16_CELL_SIZE..].iter().any(|&byte| byte != 0) {
            panic!("User public value has non-zero high bytes");
        }
        user_public_values.extend_from_slice(&bytes[..U16_CELL_SIZE]);
    }
    let published_bytes = usize::try_from(num_public_values)
        .expect("u32 fits usize")
        .checked_mul(PUBLIC_VALUE_BYTES)
        .expect("User public values length overflow");
    if published_bytes > user_public_values.len() {
        panic!("User public values count exceeds configured capacity");
    }
    if user_public_values[published_bytes..]
        .iter()
        .any(|&byte| byte != 0)
    {
        panic!("User public values padding is non-zero");
    }
    user_public_values.truncate(published_bytes);
    (num_public_values, user_public_values)
}

#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::decode_user_public_values;

    #[test]
    fn decode_user_public_values_returns_only_the_published_prefix() {
        let expanded = [
            1, 0, 0, 0, // one u64 value
            0, 0, 0, 0, // reserved
            0x34, 0x12, 0, 0, 0xcd, 0xab, 0, 0, 0x78, 0x56, 0, 0, 0xef, 0xbe, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, // padding value
        ];
        assert_eq!(
            decode_user_public_values(&expanded),
            (1, vec![0x34, 0x12, 0xcd, 0xab, 0x78, 0x56, 0xef, 0xbe])
        );
    }
}
