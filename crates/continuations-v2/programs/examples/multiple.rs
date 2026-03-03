#![cfg_attr(target_os = "zkvm", no_main)]
#![cfg_attr(target_os = "zkvm", no_std)]

use openvm_deferral_guest::{deferred_compute, get_deferred_output, Commit};

openvm::entry!(main);

const INPUT_COMMIT_0: Commit = [0x11; 32];
const INPUT_COMMIT_1: Commit = [0x22; 32];
const INPUT_COMMIT_2: Commit = [0x33; 32];
const EXPECTED_OUTPUT_0_IDX1: [u8; 8] = [2, 4, 7, 11, 16, 22, 29, 37];
const EXPECTED_OUTPUT_1_IDX1: [u8; 8] = [9, 16, 22, 27, 31, 34, 36, 37];
const EXPECTED_OUTPUT_2_IDX1: [u8; 8] = [10, 19, 28, 37, 46, 55, 64, 73];
const EXPECTED_OUTPUT_0_IDX2: [u8; 8] = [3, 5, 8, 12, 17, 23, 30, 38];

pub fn main() {
    let output_key_0 = deferred_compute::<1>(&INPUT_COMMIT_0);
    let mut output_0 = [0u8; EXPECTED_OUTPUT_0_IDX1.len()];
    get_deferred_output::<1>(&mut output_0, &output_key_0);
    assert_eq!(output_0, EXPECTED_OUTPUT_0_IDX1);

    let output_key_1 = deferred_compute::<1>(&INPUT_COMMIT_1);
    let mut output_1 = [0u8; EXPECTED_OUTPUT_1_IDX1.len()];
    get_deferred_output::<1>(&mut output_1, &output_key_1);
    assert_eq!(output_1, EXPECTED_OUTPUT_1_IDX1);

    let output_key_2 = deferred_compute::<1>(&INPUT_COMMIT_2);
    let mut output_2 = [0u8; EXPECTED_OUTPUT_2_IDX1.len()];
    get_deferred_output::<1>(&mut output_2, &output_key_2);
    assert_eq!(output_2, EXPECTED_OUTPUT_2_IDX1);

    let output_key_3 = deferred_compute::<2>(&INPUT_COMMIT_0);
    let mut output_3 = [0u8; EXPECTED_OUTPUT_0_IDX2.len()];
    get_deferred_output::<2>(&mut output_3, &output_key_3);
    assert_eq!(output_3, EXPECTED_OUTPUT_0_IDX2);
}
