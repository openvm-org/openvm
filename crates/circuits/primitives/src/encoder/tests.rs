use std::sync::Arc;

use openvm_cuda_backend::{
    base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix, types::F,
};
use openvm_stark_backend::{p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix};

use crate::{cuda_abi::encoder, encoder::Encoder};

#[test]
fn test_cuda_encoder_with_invalid_row() {
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
        encoder::dummy_tracegen(
            gpu_matrix.buffer(),
            num_flags as u32,
            max_degree,
            reserve_invalid,
            expected_k as u32,
        )
        .unwrap();
    };

    assert_eq_host_and_device_matrix(cpu_matrix, &gpu_matrix);
}

#[test]
fn test_cuda_encoder_without_invalid_row() {
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
        encoder::dummy_tracegen(
            gpu_matrix.buffer(),
            num_flags as u32,
            max_degree,
            reserve_invalid,
            expected_k as u32,
        )
        .unwrap();
    };

    assert_eq_host_and_device_matrix(cpu_matrix, &gpu_matrix);
}
