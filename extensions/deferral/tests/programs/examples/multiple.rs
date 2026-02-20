#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_deferral_guest::{
    compute_deferral_output, get_deferral_output, setup_deferrals, Commit,
};

openvm::entry!(main);

const INPUT_COMMIT_0: Commit = [0x11; 32];
const INPUT_COMMIT_1: Commit = [0x22; 32];
const INPUT_COMMIT_2: Commit = [0x33; 32];
const EXPECTED_OUTPUT_IDX0_COMMIT0: [u8; 8] = [1, 3, 6, 10, 15, 21, 28, 36];
const EXPECTED_OUTPUT_IDX0_COMMIT1: [u8; 8] = [8, 15, 21, 26, 30, 33, 35, 36];
const EXPECTED_OUTPUT_IDX0_COMMIT2: [u8; 8] = [9, 18, 27, 36, 45, 54, 63, 72];
const EXPECTED_OUTPUT_IDX1_COMMIT0: [u8; 8] = [2, 4, 7, 11, 16, 22, 29, 37];

pub fn main() {
    setup_deferrals();

    let output_key_0 = compute_deferral_output::<0>(&INPUT_COMMIT_0);
    let mut output_0 = [0u8; EXPECTED_OUTPUT_IDX0_COMMIT0.len()];
    get_deferral_output::<0>(&mut output_0, &output_key_0);
    assert_eq!(output_0, EXPECTED_OUTPUT_IDX0_COMMIT0);

    let output_key_1 = compute_deferral_output::<0>(&INPUT_COMMIT_1);
    let mut output_1 = [0u8; EXPECTED_OUTPUT_IDX0_COMMIT1.len()];
    get_deferral_output::<0>(&mut output_1, &output_key_1);
    assert_eq!(output_1, EXPECTED_OUTPUT_IDX0_COMMIT1);

    let output_key_2 = compute_deferral_output::<0>(&INPUT_COMMIT_2);
    let mut output_2 = [0u8; EXPECTED_OUTPUT_IDX0_COMMIT2.len()];
    get_deferral_output::<0>(&mut output_2, &output_key_2);
    assert_eq!(output_2, EXPECTED_OUTPUT_IDX0_COMMIT2);

    let output_key_3 = compute_deferral_output::<1>(&INPUT_COMMIT_0);
    let mut output_3 = [0u8; EXPECTED_OUTPUT_IDX1_COMMIT0.len()];
    get_deferral_output::<1>(&mut output_3, &output_key_3);
    assert_eq!(output_3, EXPECTED_OUTPUT_IDX1_COMMIT0);
}
