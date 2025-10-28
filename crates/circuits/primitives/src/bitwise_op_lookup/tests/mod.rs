use std::{iter, sync::Arc};

use dummy::DummyAir;
use openvm_stark_backend::{
    p3_field::FieldAlgebra,
    p3_matrix::dense::RowMajorMatrix,
    p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
    utils::disable_debug_builder,
    verifier::VerificationError,
    AirRef,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::Rng;
#[cfg(feature = "cuda")]
use {
    crate::bitwise_op_lookup::{BitwiseOperationLookupAir, BitwiseOperationLookupChipGPU},
    dummy::cuda::DummyInteractionChipGPU,
    openvm_cuda_backend::{
        base::DeviceMatrix,
        engine::GpuBabyBearPoseidon2Engine,
        types::{F, SC},
    },
    openvm_cuda_common::copy::MemCopyH2D as _,
    openvm_stark_backend::{p3_air::BaseAir, prover::types::AirProvingContext, Chip},
    openvm_stark_sdk::{
        config::FriParameters, dummy_airs::interaction::dummy_interaction_air::DummyInteractionAir,
    },
};

use crate::bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip};

mod dummy;

const NUM_BITS: usize = 4;
const LIST_LEN: usize = 1 << 8;

#[derive(Clone, Copy)]
enum BitwiseOperation {
    Range = 0,
    Xor = 1,
}

fn generate_rng_values(
    num_lists: usize,
    list_len: usize,
) -> Vec<Vec<(u32, u32, u32, BitwiseOperation)>> {
    let mut rng = create_seeded_rng();
    (0..num_lists)
        .map(|_| {
            (0..list_len)
                .map(|_| {
                    let op = match rng.gen_range(0..2) {
                        0 => BitwiseOperation::Range,
                        _ => BitwiseOperation::Xor,
                    };
                    let x = rng.gen_range(0..(1 << NUM_BITS));
                    let y = rng.gen_range(0..(1 << NUM_BITS));
                    let z = match op {
                        BitwiseOperation::Range => 0,
                        BitwiseOperation::Xor => x ^ y,
                    };
                    (x, y, z, op)
                })
                .collect::<Vec<(u32, u32, u32, BitwiseOperation)>>()
        })
        .collect::<Vec<Vec<(u32, u32, u32, BitwiseOperation)>>>()
}

#[test]
fn test_bitwise_operation_lookup() {
    const NUM_LISTS: usize = 10;

    let bus = BitwiseOperationLookupBus::new(0);
    let lookup = BitwiseOperationLookupChip::<NUM_BITS>::new(bus);

    let lists: Vec<Vec<(u32, u32, u32, BitwiseOperation)>> =
        generate_rng_values(NUM_LISTS, LIST_LEN);

    let dummies = (0..NUM_LISTS)
        .map(|_| DummyAir::new(bus))
        .collect::<Vec<_>>();

    let chips = dummies
        .into_iter()
        .map(|list| Arc::new(list) as AirRef<_>)
        .chain(iter::once(Arc::new(lookup.air) as AirRef<_>))
        .collect::<Vec<AirRef<_>>>();

    let mut traces = lists
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.iter()
                    .flat_map(|&(x, y, z, op)| {
                        match op {
                            BitwiseOperation::Range => lookup.request_range(x, y),
                            BitwiseOperation::Xor => {
                                lookup.request_xor(x, y);
                            }
                        };
                        [x, y, z, op as u32].into_iter()
                    })
                    .map(FieldAlgebra::from_canonical_u32)
                    .collect(),
                4,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();
    traces.push(lookup.generate_trace());

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(chips, traces)
        .expect("Verification failed");
}

fn run_negative_test(bad_row: (u32, u32, u32, BitwiseOperation)) {
    let bus = BitwiseOperationLookupBus::new(0);
    let lookup = BitwiseOperationLookupChip::<NUM_BITS>::new(bus);

    let mut list = generate_rng_values(1, LIST_LEN - 1)[0].clone();
    list.push(bad_row);

    let dummy = DummyAir::new(bus);
    let chips = any_rap_arc_vec![dummy, lookup.air];

    let traces = vec![
        RowMajorMatrix::new(
            list.iter()
                .flat_map(|&(x, y, z, op)| {
                    match op {
                        BitwiseOperation::Range => lookup.request_range(x, y),
                        BitwiseOperation::Xor => {
                            lookup.request_xor(x, y);
                        }
                    };
                    [x, y, z, op as u32].into_iter()
                })
                .map(FieldAlgebra::from_canonical_u32)
                .collect(),
            4,
        ),
        lookup.generate_trace(),
    ];

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(chips, traces).err(),
        Some(VerificationError::ChallengePhaseError),
        "Expected constraint to fail"
    );
}

#[test]
fn negative_test_bitwise_operation_lookup_range_wrong_z() {
    run_negative_test((2, 1, 1, BitwiseOperation::Range));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_range_x_out_of_range() {
    run_negative_test((16, 1, 0, BitwiseOperation::Range));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_range_y_out_of_range() {
    run_negative_test((1, 16, 0, BitwiseOperation::Range));
}

#[test]
fn negative_test_bitwise_operation_lookup_xor_wrong_z() {
    // 1011(11) ^ 0101(5) = 1110(14)
    run_negative_test((11, 5, 15, BitwiseOperation::Xor));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_xor_x_out_of_range() {
    // 10000(16) ^ 0001(1) = 0001(1) in 4 bits, but need x < 2^NUM_BITS
    run_negative_test((16, 1, 1, BitwiseOperation::Xor));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_xor_y_out_of_range() {
    // 0001(1) ^ 10000(16) = 0001(1) in 4 bits, but need y < 2^NUM_BITS
    run_negative_test((1, 16, 1, BitwiseOperation::Xor));
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitwise_op_lookup() {
    const CUDA_NUM_BITS: usize = 8;
    const NUM_INPUTS: usize = 1 << 16;
    const BIT_MASK: u32 = (1 << CUDA_NUM_BITS) - 1;

    let mut rng = create_seeded_rng();
    let bitwise = Arc::new(BitwiseOperationLookupChipGPU::<CUDA_NUM_BITS>::new());

    let random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            let x = rng.gen::<u32>() & BIT_MASK;
            let y = rng.gen::<u32>() & BIT_MASK;
            let op = rng.gen_bool(0.5);
            [x, y, op as u32]
        })
        .collect::<Vec<_>>();
    let dummy_chip = DummyInteractionChipGPU::new(bitwise.clone(), random_values);

    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(DummyInteractionAir::new(4, true, 0)),
        Arc::new(BitwiseOperationLookupAir::<CUDA_NUM_BITS>::new(
            BitwiseOperationLookupBus::new(0),
        )),
    ];
    let dummy_ctx = dummy_chip.generate_proving_ctx(());
    let bitwise_ctx = bitwise.generate_proving_ctx(());
    let ctxs = vec![dummy_ctx, bitwise_ctx];

    let engine: GpuBabyBearPoseidon2Engine =
        GpuBabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1));
    engine.run_test(airs, ctxs).expect("Verification failed");
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_bitwise_op_lookup_hybrid() {
    const CUDA_NUM_BITS: usize = 8;
    const NUM_INPUTS: usize = 1 << 16;
    const BIT_MASK: u32 = (1 << CUDA_NUM_BITS) - 1;

    let mut rng = create_seeded_rng();
    let bus = BitwiseOperationLookupBus::new(0);
    let bitwise = Arc::new(BitwiseOperationLookupChipGPU::<CUDA_NUM_BITS>::hybrid(
        Arc::new(BitwiseOperationLookupChip::new(bus)),
    ));

    let gpu_random_values = (0..NUM_INPUTS)
        .flat_map(|_| {
            let x = rng.gen::<u32>() & BIT_MASK;
            let y = rng.gen::<u32>() & BIT_MASK;
            let op = rng.gen_bool(0.5);
            [x, y, op as u32]
        })
        .collect::<Vec<_>>();
    let gpu_dummy_chip = DummyInteractionChipGPU::new(bitwise.clone(), gpu_random_values);

    let cpu_chip = bitwise.cpu_chip.clone().unwrap();
    let cpu_values = (0..NUM_INPUTS)
        .map(|_| {
            let x = rng.gen::<u32>() & BIT_MASK;
            let y = rng.gen::<u32>() & BIT_MASK;
            let op_xor = rng.gen_bool(0.5);
            let z = if op_xor {
                cpu_chip.request_xor(x, y)
            } else {
                cpu_chip.request_range(x, y);
                0
            };
            [x, y, z, op_xor as u32]
        })
        .collect::<Vec<_>>();
    let cpu_dummy_trace = (0..NUM_INPUTS)
        .map(|_| F::ONE)
        .chain(
            cpu_values
                .iter()
                .map(|v| F::from_canonical_u32(v[0]))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[1])))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[2])))
                .chain(cpu_values.iter().map(|v| F::from_canonical_u32(v[3]))),
        )
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let dummy_air = DummyInteractionAir::new(4, true, bus.inner.index);
    let cpu_proving_ctx = AirProvingContext {
        cached_mains: vec![],
        common_main: Some(DeviceMatrix::new(
            Arc::new(cpu_dummy_trace),
            NUM_INPUTS,
            BaseAir::<F>::width(&dummy_air),
        )),
        public_values: vec![],
    };

    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(dummy_air),
        Arc::new(dummy_air),
        Arc::new(BitwiseOperationLookupAir::<CUDA_NUM_BITS>::new(bus)),
    ];
    let ctxs = vec![
        cpu_proving_ctx,
        gpu_dummy_chip.generate_proving_ctx(()),
        bitwise.generate_proving_ctx(()),
    ];

    let engine = GpuBabyBearPoseidon2Engine::new(FriParameters::new_for_testing(1));
    engine.run_test(airs, ctxs).expect("Verification failed");
}
