use core::borrow::Borrow;

use openvm_stark_sdk::config::setup_tracing_with_log_level;
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;
use stark_backend_v2::F;
use tracing::Level;

use super::{
    air::ExpBitsLenCols,
    trace::{ExpBitsLenCpuTraceGenerator, NUM_BITS_MAX_PLUS_ONE, fill_valid_rows},
};

#[derive(Clone, Debug)]
pub(crate) struct ExpBitsLenTestRequest {
    pub num_bits: u8,
    pub base: u32,
    pub bit_src: u32,
}

pub(crate) fn sample_exp_bits_len_requests(num_requests: usize) -> Vec<ExpBitsLenTestRequest> {
    (0..num_requests)
        .map(|idx| {
            let idx_u32 = idx as u32;
            let modulus = F::ORDER_U32;
            let base = idx_u32
                .wrapping_mul(0x045d_9f3b)
                .wrapping_add(5 * ((idx_u32 % 19) + 1))
                % modulus;
            let bit_src = idx_u32
                .wrapping_mul(0x9e37_79b1)
                .wrapping_add(0xA5A5_5A5A)
                .rotate_left((idx % 5) as u32)
                % modulus;
            let num_bits = (idx % NUM_BITS_MAX_PLUS_ONE) as u8;
            ExpBitsLenTestRequest {
                base,
                bit_src,
                num_bits,
            }
        })
        .collect()
}

#[test_case::test_case(1)]
#[test_case::test_case(1_000)]
#[test_case::test_case(1_000_000)]
fn test_exp_bits_len_cpu_trace_generation(num_requests: usize) {
    setup_tracing_with_log_level(Level::DEBUG);

    let requests = sample_exp_bits_len_requests(num_requests);
    let generator = ExpBitsLenCpuTraceGenerator::default();
    generator.add_requests(requests.iter().map(|req| {
        (
            F::from_u32(req.base),
            F::from_u32(req.bit_src),
            req.num_bits as usize,
        )
    }));

    let trace = generator.generate_trace_row_major();
    let width = ExpBitsLenCols::<F>::width();
    assert_eq!(trace.width(), width);

    let total_valid_rows: usize = requests.iter().map(|req| req.num_bits as usize + 1).sum();
    assert_eq!(trace.height(), total_valid_rows.next_power_of_two());

    let mut cursor = 0;
    for req in &requests {
        let row_count = req.num_bits as usize + 1;
        let start = cursor * width;
        let end = start + row_count * width;
        let mut expected = vec![F::ZERO; row_count * width];
        fill_valid_rows(
            F::from_u32(req.base),
            req.bit_src,
            req.num_bits,
            &mut expected,
            width,
        );
        assert_eq!(&trace.values[start..end], &expected);
        cursor += row_count;
    }

    assert_eq!(cursor, total_valid_rows);

    for row_idx in cursor..trace.height() {
        let row_slice = &trace.values[row_idx * width..(row_idx + 1) * width];
        let cols: &ExpBitsLenCols<F> = row_slice.borrow();
        assert_eq!(cols.is_valid, F::ZERO);
        assert_eq!(cols.result, F::ONE);
    }
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use std::sync::Arc;

    use openvm_cuda_backend::data_transporter::assert_eq_host_and_device_matrix;

    use super::*;
    use crate::primitives::exp_bits_len::ExpBitsLenGpuTraceGenerator;

    #[test_case::test_case(1)]
    #[test_case::test_case(1_000)]
    #[test_case::test_case(1_000_000)]
    fn test_exp_bits_len_gpu_trace_generation(num_requests: usize) {
        setup_tracing_with_log_level(Level::DEBUG);

        let requests = sample_exp_bits_len_requests(num_requests);

        let cpu_trace = {
            let cpu_gen = ExpBitsLenCpuTraceGenerator::default();
            cpu_gen.add_requests(requests.iter().map(|req| {
                (
                    F::from_u32(req.base),
                    F::from_u32(req.bit_src),
                    req.num_bits as usize,
                )
            }));
            cpu_gen.generate_trace_row_major()
        };

        let gpu_trace = {
            let gpu_gen = ExpBitsLenGpuTraceGenerator::default();
            gpu_gen.add_requests(requests.iter().map(|req| {
                (
                    F::from_u32(req.base),
                    F::from_u32(req.bit_src),
                    req.num_bits as usize,
                )
            }));
            gpu_gen.generate_trace_device()
        };

        assert_eq_host_and_device_matrix(Arc::new(cpu_trace), &gpu_trace);
    }
}
