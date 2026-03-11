use core::borrow::Borrow;
use std::borrow::BorrowMut;

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    any_air_arc_vec,
    interaction::InteractionBuilder,
    prover::{
        AirProvingContext, ColMajorMatrix, CpuBackend, DeviceDataTransporter, ProvingContext,
    },
    utils::disable_debug_builder,
    verifier::VerifierError,
    BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine, StarkProtocolConfig,
};
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F},
    utils::setup_tracing_with_log_level,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use tracing::Level;

use super::{
    air::{ExpBitsLenAir, ExpBitsLenCols},
    trace::{
        fill_valid_rows, fill_valid_rows_with_decomp_src, ExpBitsLenCpuTraceGenerator,
        NUM_BITS_MAX_PLUS_ONE,
    },
};
use crate::{
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage, RightShiftBus},
    tests::test_engine_small,
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

#[repr(C)]
#[derive(AlignedBorrow)]
struct ExpBitsLenLookupCols<T> {
    enabled: T,
    base: T,
    bit_src: T,
    num_bits: T,
    result: T,
}

#[derive(Clone, Copy, Debug)]
struct ExpBitsLenLookupAir {
    exp_bits_len_bus: ExpBitsLenBus,
}

impl<F> BaseAir<F> for ExpBitsLenLookupAir {
    fn width(&self) -> usize {
        ExpBitsLenLookupCols::<F>::width()
    }
}

impl<F> BaseAirWithPublicValues<F> for ExpBitsLenLookupAir {}
impl<F> PartitionedBaseAir<F> for ExpBitsLenLookupAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ExpBitsLenLookupAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should contain at least one row");
        let local: &ExpBitsLenLookupCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.enabled);
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.base,
                bit_src: local.bit_src,
                num_bits: local.num_bits,
                result: local.result,
            },
            local.enabled,
        );
    }
}

fn build_lookup_trace(base: F, bit_src: F, num_bits: usize, result: F) -> RowMajorMatrix<F> {
    let width = ExpBitsLenLookupCols::<F>::width();
    let mut values = vec![F::ZERO; 2 * width];
    let row = &mut values[..width];
    let cols: &mut ExpBitsLenLookupCols<F> = row.borrow_mut();
    cols.enabled = F::ONE;
    cols.base = base;
    cols.bit_src = bit_src;
    cols.num_bits = F::from_usize(num_bits);
    cols.result = result;
    RowMajorMatrix::new(values, width)
}

fn pow_by_low_bits(base: F, bit_src: u32, num_bits: usize) -> F {
    let exponent = if num_bits == 0 {
        0
    } else {
        bit_src & ((1u32 << num_bits) - 1)
    };
    let mut acc = F::ONE;
    for _ in 0..exponent {
        acc *= base;
    }
    acc
}

fn add_terminal_tail(exp_bits_trace: &mut RowMajorMatrix<F>, delta: F) {
    let width = exp_bits_trace.width();
    let mut tail = delta;
    for step in (0..NUM_BITS_MAX_PLUS_ONE).rev() {
        let row_slice = &mut exp_bits_trace.values[step * width..(step + 1) * width];
        let cols: &mut ExpBitsLenCols<F> = row_slice.borrow_mut();
        cols.bit_src += tail;
        tail += tail;
    }
}

fn prove_and_verify_exp_bits(
    lookup_trace: RowMajorMatrix<F>,
    exp_bits_trace: RowMajorMatrix<F>,
) -> Result<(), VerifierError<<BabyBearPoseidon2Config as StarkProtocolConfig>::EF>> {
    disable_debug_builder();

    let exp_bits_len_bus = ExpBitsLenBus::new(0);
    let right_shift_bus = RightShiftBus::new(1);
    let airs = any_air_arc_vec![
        ExpBitsLenLookupAir { exp_bits_len_bus },
        ExpBitsLenAir::new(exp_bits_len_bus, right_shift_bus)
    ];

    let engine = test_engine_small();
    let (pk, vk) = engine.keygen(&airs);
    let ctx: ProvingContext<CpuBackend<BabyBearPoseidon2Config>> = ProvingContext::new(
        [lookup_trace, exp_bits_trace]
            .into_iter()
            .enumerate()
            .map(|(air_idx, trace)| {
                (
                    air_idx,
                    AirProvingContext::<CpuBackend<BabyBearPoseidon2Config>>::simple_no_pis(
                        ColMajorMatrix::from_row_major(&trace),
                    ),
                )
            })
            .collect(),
    );

    let device = engine.device();
    let d_pk = device.transport_pk_to_device(&pk);
    let d_ctx: ProvingContext<CpuBackend<BabyBearPoseidon2Config>> = ProvingContext::new(
        ctx.into_iter()
            .map(|(air_idx, air_ctx)| {
                (
                    air_idx,
                    AirProvingContext::<CpuBackend<BabyBearPoseidon2Config>> {
                        cached_mains: vec![],
                        common_main: device.transport_matrix_to_device(&air_ctx.common_main),
                        public_values: air_ctx.public_values,
                    },
                )
            })
            .collect(),
    );

    let proof = engine.prove(&d_pk, d_ctx).unwrap();
    engine.verify(&vk, &proof)
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

    let trace = generator
        .generate_trace_row_major(None)
        .expect("trace height should be unconstrained");
    let width = ExpBitsLenCols::<F>::width();
    assert_eq!(trace.width(), width);

    let total_valid_rows = requests.len() * NUM_BITS_MAX_PLUS_ONE;
    assert_eq!(trace.height(), total_valid_rows.next_power_of_two());

    let mut cursor = 0;
    for req in &requests {
        let row_count = NUM_BITS_MAX_PLUS_ONE;
        let start = cursor * width;
        let end = start + row_count * width;
        let mut expected = vec![F::ZERO; row_count * width];
        fill_valid_rows(
            F::from_u32(req.base),
            req.bit_src,
            req.num_bits,
            0,
            0,
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

#[test]
fn test_exp_bits_len_proves_honest_trace() {
    let base = F::from_u32(3);
    let bit_src = F::from_u32(45);
    let num_bits = 4usize;
    let result = pow_by_low_bits(base, bit_src.as_canonical_u32(), num_bits);

    let exp_bits_gen = ExpBitsLenCpuTraceGenerator::default();
    exp_bits_gen.add_request(base, bit_src, num_bits);

    let lookup_trace = build_lookup_trace(base, bit_src, num_bits, result);
    let exp_bits_trace = exp_bits_gen
        .generate_trace_row_major(None)
        .expect("exp_bits_len trace should fit");

    prove_and_verify_exp_bits(lookup_trace, exp_bits_trace).unwrap();
}

#[test]
fn test_exp_bits_len_rejects_noncanonical_31_bit_decomposition() {
    let base = F::from_u32(3);
    let bit_src = F::from_u32(2);
    let num_bits = 2usize;
    let forged_decomp_src = F::ORDER_U32 + 2;
    let forged_result = pow_by_low_bits(base, forged_decomp_src, num_bits);

    let lookup_trace = build_lookup_trace(base, bit_src, num_bits, forged_result);

    let width = ExpBitsLenCols::<F>::width();
    let mut exp_bits_values = vec![F::ZERO; NUM_BITS_MAX_PLUS_ONE * width];
    fill_valid_rows_with_decomp_src(
        base,
        forged_decomp_src,
        num_bits as u8,
        0,
        0,
        &mut exp_bits_values,
        width,
    );
    let exp_bits_trace = RowMajorMatrix::new(exp_bits_values, width);

    let result = prove_and_verify_exp_bits(lookup_trace, exp_bits_trace);
    assert!(result.is_err());
}

#[test]
fn test_exp_bits_len_rejects_nonzero_terminal_tail_on_last_row() {
    let base = F::from_u32(3);
    let bit_src = F::from_u32(45);
    let num_bits = 4usize;
    let honest_result = pow_by_low_bits(base, bit_src.as_canonical_u32(), num_bits);

    let exp_bits_gen = ExpBitsLenCpuTraceGenerator::default();
    exp_bits_gen.add_request(base, bit_src, num_bits);
    let mut exp_bits_trace = exp_bits_gen
        .generate_trace_row_major(Some(NUM_BITS_MAX_PLUS_ONE))
        .expect("single request should fit exactly");

    add_terminal_tail(&mut exp_bits_trace, F::ONE);

    let first_row = &exp_bits_trace.values[..exp_bits_trace.width()];
    let first_cols: &ExpBitsLenCols<F> = first_row.borrow();
    let forged_bit_src = first_cols.bit_src;
    assert_ne!(
        pow_by_low_bits(base, forged_bit_src.as_canonical_u32(), num_bits),
        honest_result
    );

    let lookup_trace = build_lookup_trace(base, forged_bit_src, num_bits, honest_result);
    let result = prove_and_verify_exp_bits(lookup_trace, exp_bits_trace);
    assert!(result.is_err());
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
            cpu_gen
                .generate_trace_row_major(None)
                .expect("trace height should be unconstrained")
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
            gpu_gen
                .generate_trace_device(None)
                .expect("trace height should be unconstrained")
        };

        assert_eq_host_and_device_matrix(Arc::new(cpu_trace), &gpu_trace);
    }
}
