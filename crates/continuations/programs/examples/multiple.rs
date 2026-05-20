#![cfg_attr(
    all(not(feature = "std"), any(openvm_intrinsics, target_os = "openvm")),
    no_main
)]
#![cfg_attr(not(feature = "std"), no_std)]

use openvm::io::read;
use openvm_deferral_guest::{deferred_compute, get_deferred_output, Commit};

openvm::entry!(main);

const EXPECTED_OUTPUT_0_IDX1: [u8; 16] = [
    2, 4, 7, 11, 16, 22, 29, 37, 46, 56, 67, 79, 92, 106, 121, 137,
];
const EXPECTED_OUTPUT_1_IDX1: [u8; 16] = [
    17, 32, 46, 59, 71, 82, 92, 101, 109, 116, 122, 127, 131, 134, 136, 137,
];
const EXPECTED_OUTPUT_2_IDX1: [u8; 16] = [
    10, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 109, 118, 127, 136, 145,
];
const EXPECTED_OUTPUT_0_IDX2: [u8; 16] = [
    3, 5, 8, 12, 17, 23, 30, 38, 47, 57, 68, 80, 93, 107, 122, 138,
];

pub fn main() {
    let input_commit_0: Commit = read();
    let output_key_0 = deferred_compute::<1>(&input_commit_0);
    let mut output_0 = [0u8; EXPECTED_OUTPUT_0_IDX1.len()];
    get_deferred_output::<1>(&mut output_0, &output_key_0);
    assert_eq!(output_0, EXPECTED_OUTPUT_0_IDX1);

    let input_commit_1: Commit = read();
    let output_key_1 = deferred_compute::<1>(&input_commit_1);
    let mut output_1 = [0u8; EXPECTED_OUTPUT_1_IDX1.len()];
    get_deferred_output::<1>(&mut output_1, &output_key_1);
    assert_eq!(output_1, EXPECTED_OUTPUT_1_IDX1);

    let input_commit_2: Commit = read();
    let output_key_2 = deferred_compute::<1>(&input_commit_2);
    let mut output_2 = [0u8; EXPECTED_OUTPUT_2_IDX1.len()];
    get_deferred_output::<1>(&mut output_2, &output_key_2);
    assert_eq!(output_2, EXPECTED_OUTPUT_2_IDX1);

    let output_key_3 = deferred_compute::<2>(&input_commit_0);
    let mut output_3 = [0u8; EXPECTED_OUTPUT_0_IDX2.len()];
    get_deferred_output::<2>(&mut output_3, &output_key_3);
    assert_eq!(output_3, EXPECTED_OUTPUT_0_IDX2);
}
