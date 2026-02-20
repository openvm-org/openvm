#![cfg_attr(not(feature = "std"), no_main)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm_deferral_guest::{
    compute_deferral_output, get_deferral_output, setup_deferrals, Commit,
};

openvm::entry!(main);

const INPUT_COMMIT_0: Commit = [0x11; 32];
const INPUT_COMMIT_1: Commit = [0x22; 32];
const INPUT_COMMIT_2: Commit = [0x33; 32];
const EXPECTED_OUTPUT_0: [u8; 8] = [1, 3, 6, 10, 15, 21, 28, 36];
const EXPECTED_OUTPUT_1: [u8; 8] = [8, 15, 21, 26, 30, 33, 35, 36];
const EXPECTED_OUTPUT_2: [u8; 8] = [9, 18, 27, 36, 45, 54, 63, 72];

pub fn main() {
    setup_deferrals();

    let output_key_0 = compute_deferral_output::<0>(&INPUT_COMMIT_0);
    assert_eq!(output_key_0.output_len() as usize, EXPECTED_OUTPUT_0.len());

    let mut output_0 = [0u8; EXPECTED_OUTPUT_0.len()];
    get_deferral_output::<0>(&mut output_0, &output_key_0);
    assert_eq!(output_0, EXPECTED_OUTPUT_0);

    let output_key_1 = compute_deferral_output::<0>(&INPUT_COMMIT_1);
    assert_eq!(output_key_1.output_len() as usize, EXPECTED_OUTPUT_1.len());

    let mut output_1 = [0u8; EXPECTED_OUTPUT_1.len()];
    get_deferral_output::<0>(&mut output_1, &output_key_1);
    assert_eq!(output_1, EXPECTED_OUTPUT_1);

    let output_key_2 = compute_deferral_output::<0>(&INPUT_COMMIT_2);
    assert_eq!(output_key_2.output_len() as usize, EXPECTED_OUTPUT_2.len());

    let mut output_2 = [0u8; EXPECTED_OUTPUT_2.len()];
    get_deferral_output::<0>(&mut output_2, &output_key_2);
    assert_eq!(output_2, EXPECTED_OUTPUT_2);
}
