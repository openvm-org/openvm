use std::sync::Arc;

use cuda_kernels::dummy::{
    fibair::fibair_tracegen,
    is_zero,
    less_than::{assert_less_than_tracegen, less_than_array_tracegen, less_than_tracegen},
};
use cuda_utils::copy::MemCopyD2H;
use cuda_utils::{copy::MemCopyH2D, d_buffer::DeviceBuffer};
use openvm_circuit_primitives::{is_zero::IsZeroSubAir, TraceSubRowGenerator};
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;
use openvm_stark_sdk::{dummy_airs::fib_air::chip::FibonacciChip, utils::create_seeded_rng};
use p3_baby_bear::BabyBear;
use p3_field::{FieldAlgebra, PrimeField32};
use rand::Rng;
use stark_backend_gpu::{base::DeviceMatrix, prelude::F};
use tracegen_gpu::utils::{assert_eq_cpu_and_gpu_matrix, test_chip_whole_trace_output};

#[test]
fn test_fibair_tracegen() {
    let mut rng = create_seeded_rng();
    for log_height in 1..=2 {
        let a = rng.gen_range(1..F::ORDER_U32);
        let b = 1;
        let n = 1 << log_height;

        let output = DeviceMatrix::<F>::with_capacity(n, 2);
        unsafe {
            fibair_tracegen(output.buffer(), a, b, n).unwrap();
        };

        let chip = FibonacciChip::new(a, b, n);
        test_chip_whole_trace_output(chip, &output);
    }
}

#[test]
fn test_assert_less_than_tracegen() {
    let max_bits: usize = 29;
    let decomp: usize = 8;
    const AUX_LEN: usize = 4;

    let num_pairs = 4;
    let trace = DeviceMatrix::<F>::with_capacity(num_pairs, 3 + AUX_LEN);
    let pairs = vec![[14321, 26883], [0, 1], [28, 120], [337, 456]]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let rc_num_bins = (1 << (decomp + 1)) as usize;
    let rc_histogram = DeviceBuffer::<u32>::with_capacity(rc_num_bins);

    unsafe {
        assert_less_than_tracegen(
            trace.buffer(),
            num_pairs,
            &pairs,
            max_bits,
            AUX_LEN,
            &rc_histogram,
        )
        .unwrap();
    }

    // From test_lt_chip_decomp_does_not_divide in OpenVM's assert_less_than tests
    let expected_cpu_matrix_vals: [[u32; 7]; 4] = [
        [14321, 26883, 1, 17, 49, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [28, 120, 1, 91, 0, 0, 0],
        [337, 456, 1, 118, 0, 0, 0],
    ];
    let expected_cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        expected_cpu_matrix_vals
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect(),
        3 + AUX_LEN,
    ));

    assert_eq_cpu_and_gpu_matrix(expected_cpu_matrix, &trace);
}

#[test]
fn test_less_than_tracegen() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    const AUX_LEN: usize = 2;

    let num_pairs = 4;
    let trace = DeviceMatrix::<F>::with_capacity(num_pairs, 3 + AUX_LEN);
    let pairs = vec![[14321, 26883], [1, 0], [773, 773], [337, 456]]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let rc_num_bins = (1 << (decomp + 1)) as usize;
    let rc_histogram = DeviceBuffer::<u32>::with_capacity(rc_num_bins);

    unsafe {
        less_than_tracegen(
            trace.buffer(),
            num_pairs,
            &pairs,
            max_bits,
            AUX_LEN,
            &rc_histogram,
        )
        .unwrap();
    }

    // From test_lt_chip_decomp_does_not_divide in OpenVM's is_less_than tests
    let expected_cpu_matrix_vals: [[u32; 5]; 4] = [
        [14321, 26883, 1, 17, 49],
        [1, 0, 0, 254, 255],
        [773, 773, 0, 255, 255],
        [337, 456, 1, 118, 0],
    ];
    let expected_cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        expected_cpu_matrix_vals
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect(),
        3 + AUX_LEN,
    ));

    assert_eq_cpu_and_gpu_matrix(expected_cpu_matrix, &trace);
}

#[test]
fn test_less_than_array_tracegen() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    const ARRAY_LEN: usize = 2;
    const AUX_LEN: usize = 2;

    let num_pairs = 4;
    let trace = DeviceMatrix::<F>::with_capacity(num_pairs, 3 * ARRAY_LEN + AUX_LEN + 2);
    let pairs = vec![
        [14321, 123, 26678, 233],
        [26678, 244, 14321, 233],
        [14321, 244, 14321, 244],
        [14321, 233, 14321, 244],
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .to_device()
    .unwrap();

    let rc_num_bins = (1 << (decomp + 1)) as usize;
    let rc_histogram = DeviceBuffer::<u32>::with_capacity(rc_num_bins);

    unsafe {
        less_than_array_tracegen(
            trace.buffer(),
            num_pairs,
            &pairs,
            max_bits,
            ARRAY_LEN,
            AUX_LEN,
            &rc_histogram,
        )
        .unwrap();
    }

    // From test_is_less_than_tuple_chip in OpenVM's is_less_than_array tests, modified slightly
    let expected_cpu_matrix_vals = [
        [14321, 123, 26678, 233, 1, 1, 0, 1344947008, 68, 48],
        [26678, 244, 14321, 233, 0, 1, 0, 668318913, 186, 207],
        [14321, 244, 14321, 244, 0, 0, 0, 0, 255, 255],
        [14321, 233, 14321, 244, 1, 0, 1, 549072524, 10, 0],
    ];
    let expected_cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        expected_cpu_matrix_vals
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect(),
        3 * ARRAY_LEN + AUX_LEN + 2,
    ));

    assert_eq_cpu_and_gpu_matrix(expected_cpu_matrix, &trace);
}

#[test]
fn test_is_zero_against_cpu_full() {
    let mut rng = create_seeded_rng();
    for log_height in 1..=16 {
        let n = 1 << log_height;
        let vec_x = (0..n)
            .map(|_| {
                if rng.gen_bool(0.5) {
                    0 // 50% chance to be zero
                } else {
                    rng.gen_range(0..F::ORDER_U32) // 50% chance to be random
                }
            })
            .collect::<Vec<_>>();

        let input_buffer = vec_x.as_slice().to_device().unwrap();
        let output = DeviceMatrix::<F>::with_capacity(n, 2);
        unsafe {
            is_zero::tracegen(output.buffer(), &input_buffer).unwrap();
        };

        let results = output.to_host().unwrap();
        for i in 0..n {
            let cur_x = BabyBear::from_canonical_u32(vec_x[i]);
            let mut cur_inv = BabyBear::ZERO;
            let mut cur_out = BabyBear::ONE;
            IsZeroSubAir.generate_subrow(cur_x, (&mut cur_inv, &mut cur_out));
            assert_eq!(results[i], cur_inv);
            assert_eq!(results[n + i], cur_out);
        }
    }
}
