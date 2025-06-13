use std::sync::Arc;

use openvm_circuit_primitives::{
    encoder::Encoder, is_equal::IsEqSubAir, is_equal_array::IsEqArraySubAir, is_zero::IsZeroSubAir,
    TraceSubRowGenerator,
};
use openvm_stark_backend::p3_matrix::dense::RowMajorMatrix;
use openvm_stark_sdk::{dummy_airs::fib_air::chip::FibonacciChip, utils::create_seeded_rng};
use p3_field::{FieldAlgebra, PrimeField32};
use rand::Rng;
use stark_backend_gpu::{
    base::DeviceMatrix,
    cuda::{copy::MemCopyH2D, d_buffer::DeviceBuffer},
    prelude::F,
};
use tracegen_gpu::{
    dummy::cuda::{
        encoder,
        fibair::fibair_tracegen,
        is_equal, is_zero,
        less_than::{assert_less_than_tracegen, less_than_array_tracegen, less_than_tracegen},
    },
    utils::{assert_eq_cpu_and_gpu_matrix, test_chip_whole_trace_output},
};

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
        let vec_x: Vec<F> = (0..n)
            .map(|_| {
                if rng.gen_bool(0.5) {
                    0 // 50% chance to be zero
                } else {
                    rng.gen_range(0..F::ORDER_U32) // 50% chance to be random
                }
            })
            .map(F::from_canonical_u32)
            .collect();

        let input_buffer = vec_x.as_slice().to_device().unwrap();
        let output = DeviceMatrix::<F>::with_capacity(n, 2);
        unsafe {
            is_zero::tracegen(output.buffer(), &input_buffer).unwrap();
        };

        let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
            vec_x
                .iter()
                .flat_map(|x| {
                    let cur_x = *x;
                    let mut cur_inv = F::ZERO;
                    let mut cur_out = F::ONE;
                    IsZeroSubAir.generate_subrow(cur_x, (&mut cur_inv, &mut cur_out));
                    [cur_inv, cur_out]
                })
                .collect::<Vec<_>>(),
            2,
        ));

        assert_eq_cpu_and_gpu_matrix(cpu_matrix, &output);
    }
}

#[test]
fn test_is_equal_against_cpu_full() {
    let mut rng = create_seeded_rng();

    for log_height in 1..=16 {
        let n = 1 << log_height;

        let vec_x: Vec<F> = (0..n)
            .map(|_| F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32)))
            .collect();

        let vec_y: Vec<F> = (0..n)
            .map(|i| {
                if rng.gen_bool(0.5) {
                    vec_x[i] // 50 % chance: equal to x
                } else {
                    F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32)) // 50% chance to be random
                }
            })
            .collect();

        let inputs_x = vec_x.as_slice().to_device().unwrap();
        let inputs_y = vec_y.as_slice().to_device().unwrap();

        let gpu_matrix = DeviceMatrix::<F>::with_capacity(n, 2);
        unsafe {
            is_equal::tracegen(gpu_matrix.buffer(), &inputs_x, &inputs_y).unwrap();
        }

        let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
            (0..n)
                .flat_map(|i| {
                    let cur_x = vec_x[i];
                    let cur_y = vec_y[i];

                    let mut cur_inv = F::ONE;
                    let mut cur_out = F::ONE;
                    IsEqSubAir.generate_subrow((cur_x, cur_y), (&mut cur_inv, &mut cur_out));

                    [cur_inv, cur_out]
                })
                .collect::<Vec<_>>(),
            2,
        ));

        assert_eq_cpu_and_gpu_matrix(cpu_matrix, &gpu_matrix);
    }
}

#[test]
fn test_simple_is_equal_array_tracegen() {
    const ARRAY_LEN: usize = 4;
    let n = 4;
    let trace = DeviceMatrix::<F>::with_capacity(n, ARRAY_LEN + 1);

    let vec_x: Vec<F> = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9u32, 10, 11, 12, 13, 14, 15, 16]
        .into_iter()
        .map(F::from_canonical_u32)
        .collect();

    let vec_y: Vec<F> = vec![
        1u32, 3, 3, 4, 5, 6, 10, 8, 9u32, 10, 11, 12, 13, 200, 15, 16,
    ]
    .into_iter()
    .map(F::from_canonical_u32)
    .collect();

    let inputs_x = vec_x.as_slice().to_device().unwrap();
    let inputs_y = vec_y.as_slice().to_device().unwrap();

    unsafe { is_equal::tracegen_array(trace.buffer(), &inputs_x, &inputs_y, ARRAY_LEN).unwrap() };

    let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        (0..n)
            .flat_map(|i| {
                let cur_x: [F; ARRAY_LEN] = std::array::from_fn(|k| vec_x[i + k * n]);
                let cur_y: [F; ARRAY_LEN] = std::array::from_fn(|k| vec_y[i + k * n]);

                let mut cur_inv: [F; ARRAY_LEN] = [F::ONE; ARRAY_LEN];
                let mut cur_out = F::ONE;
                IsEqArraySubAir.generate_subrow((&cur_x, &cur_y), (&mut cur_inv, &mut cur_out));

                cur_inv.into_iter().chain(std::iter::once(cur_out))
            })
            .collect::<Vec<_>>(),
        ARRAY_LEN + 1,
    ));

    assert_eq_cpu_and_gpu_matrix(cpu_matrix, &trace);
}

#[test]
fn test_random_is_equal_array_tracegen() {
    let mut rng = create_seeded_rng();
    const ARRAY_LEN: usize = 64;

    for log_height in 1..=16 {
        let n = 1 << log_height;

        let vec_x: Vec<F> = (0..n * ARRAY_LEN)
            .map(|_| F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32)))
            .collect();

        let vec_y: Vec<F> = (0..n * ARRAY_LEN)
            .map(|_| F::from_canonical_u32(rng.gen_range(0..F::ORDER_U32)))
            .collect();

        let inputs_x = vec_x.as_slice().to_device().unwrap();
        let inputs_y = vec_y.as_slice().to_device().unwrap();

        let gpu_matrix = DeviceMatrix::<F>::with_capacity(n, ARRAY_LEN + 1);
        unsafe {
            is_equal::tracegen_array(gpu_matrix.buffer(), &inputs_x, &inputs_y, ARRAY_LEN).unwrap();
        }

        let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
            (0..n)
                .flat_map(|i| {
                    let cur_x: [F; ARRAY_LEN] = std::array::from_fn(|k| vec_x[i + k * n]);
                    let cur_y: [F; ARRAY_LEN] = std::array::from_fn(|k| vec_y[i + k * n]);

                    let mut cur_inv: [F; ARRAY_LEN] = [F::ONE; ARRAY_LEN];
                    let mut cur_out = F::ONE;
                    IsEqArraySubAir.generate_subrow((&cur_x, &cur_y), (&mut cur_inv, &mut cur_out));

                    cur_inv.into_iter().chain(std::iter::once(cur_out))
                })
                .collect::<Vec<_>>(),
            ARRAY_LEN + 1,
        ));

        assert_eq_cpu_and_gpu_matrix(cpu_matrix, &gpu_matrix);
    }
}

#[test]
fn test_encoder_with_invalid_row() {
    // Max number of flags for k = 6
    let num_flags = 461;
    let max_degree = 5;
    let reserve_invalid = true;

    let encoder = Encoder::new(num_flags, max_degree, reserve_invalid);
    let expected_k = encoder.width();

    let values = (0..num_flags)
        .map(|i| encoder.get_flag_pt(i))
        .collect::<Vec<_>>();
    let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        values
            .into_iter()
            .flat_map(|v| v.into_iter().map(F::from_canonical_u32))
            .collect(),
        expected_k,
    ));

    let gpu_matrix = DeviceMatrix::<F>::with_capacity(num_flags, expected_k);
    unsafe {
        encoder::tracegen(
            gpu_matrix.buffer(),
            num_flags as u32,
            max_degree,
            reserve_invalid,
            expected_k as u32,
        )
        .unwrap();
    };

    assert_eq_cpu_and_gpu_matrix(cpu_matrix, &gpu_matrix);
}

#[test]
fn test_encoder_without_invalid_row() {
    let num_flags = 18;
    let max_degree = 2;
    let reserve_invalid = false;

    let encoder = Encoder::new(num_flags, max_degree, reserve_invalid);
    let expected_k = encoder.width();

    let values = (0..num_flags)
        .map(|i| encoder.get_flag_pt(i))
        .collect::<Vec<_>>();
    let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        values
            .into_iter()
            .flat_map(|v| v.into_iter().map(F::from_canonical_u32))
            .collect(),
        expected_k,
    ));

    let gpu_matrix = DeviceMatrix::<F>::with_capacity(num_flags, expected_k);
    unsafe {
        encoder::tracegen(
            gpu_matrix.buffer(),
            num_flags as u32,
            max_degree,
            reserve_invalid,
            expected_k as u32,
        )
        .unwrap();
    };

    assert_eq_cpu_and_gpu_matrix(cpu_matrix, &gpu_matrix);
}
