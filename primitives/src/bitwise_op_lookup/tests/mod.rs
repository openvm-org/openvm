use std::iter;

use afs_stark_backend::{prover::USE_DEBUG_BUILDER, rap::AnyRap, verifier::VerificationError};
use ax_sdk::{
    config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    utils::create_seeded_rng,
};
use dummy::DummyAir;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use rand::Rng;

use super::BitwiseOperationLookupOpcode;
use crate::bitwise_op_lookup::{BitwiseOperationLookupBus, BitwiseOperationLookupChip};

mod dummy;

const NUM_BITS: usize = 4;
const LIST_LEN: usize = 1 << 8;

fn generate_rng_values(
    num_lists: usize,
    list_len: usize,
) -> Vec<Vec<(u32, u32, u32, BitwiseOperationLookupOpcode)>> {
    let mut rng = create_seeded_rng();
    (0..num_lists)
        .map(|_| {
            (0..list_len)
                .map(|_| {
                    let op = match rng.gen_range(0..2) {
                        0 => BitwiseOperationLookupOpcode::ADD,
                        _ => BitwiseOperationLookupOpcode::XOR,
                    };
                    let x = rng.gen_range(0..(1 << NUM_BITS));
                    let y = rng.gen_range(0..(1 << NUM_BITS));
                    let z = match op {
                        BitwiseOperationLookupOpcode::ADD => (x + y) % (1 << NUM_BITS),
                        BitwiseOperationLookupOpcode::XOR => x ^ y,
                    };
                    (x, y, z, op)
                })
                .collect::<Vec<(u32, u32, u32, BitwiseOperationLookupOpcode)>>()
        })
        .collect::<Vec<Vec<(u32, u32, u32, BitwiseOperationLookupOpcode)>>>()
}

#[test]
fn test_bitwise_operation_lookup() {
    const NUM_LISTS: usize = 10;

    let bus = BitwiseOperationLookupBus::new(0);
    let lookup = BitwiseOperationLookupChip::<NUM_BITS>::new(bus);

    let lists: Vec<Vec<(u32, u32, u32, BitwiseOperationLookupOpcode)>> =
        generate_rng_values(NUM_LISTS, LIST_LEN);

    let dummies = (0..NUM_LISTS)
        .map(|_| DummyAir::new(bus))
        .collect::<Vec<_>>();

    let chips = dummies
        .iter()
        .map(|list| list as &dyn AnyRap<_>)
        .chain(iter::once(&lookup.air as &dyn AnyRap<_>))
        .collect::<Vec<_>>();

    let mut traces = lists
        .par_iter()
        .map(|list| {
            RowMajorMatrix::new(
                list.iter()
                    .flat_map(|&(x, y, z, op)| {
                        lookup.add_count(x, y, op);
                        [x, y, z, op as u32].into_iter()
                    })
                    .map(AbstractField::from_canonical_u32)
                    .collect(),
                4,
            )
        })
        .collect::<Vec<RowMajorMatrix<BabyBear>>>();
    traces.push(lookup.generate_trace());

    BabyBearPoseidon2Engine::run_simple_test_no_pis(&chips, traces).expect("Verification failed");
}

fn run_negative_test(bad_row: (u32, u32, u32, BitwiseOperationLookupOpcode)) {
    let bus = BitwiseOperationLookupBus::new(0);
    let lookup = BitwiseOperationLookupChip::<NUM_BITS>::new(bus);

    let mut list = generate_rng_values(1, LIST_LEN - 1)[0].clone();
    list.push(bad_row);

    let dummy = DummyAir::new(bus);
    let chips = vec![&dummy as &dyn AnyRap<_>, &lookup.air];

    let traces = vec![
        RowMajorMatrix::new(
            list.iter()
                .flat_map(|&(x, y, z, op)| {
                    lookup.add_count(x, y, op);
                    [x, y, z, op as u32].into_iter()
                })
                .map(AbstractField::from_canonical_u32)
                .collect(),
            4,
        ),
        lookup.generate_trace(),
    ];

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis(&chips, traces).err(),
        Some(VerificationError::NonZeroCumulativeSum),
        "Expected constraint to fail"
    );
}

#[test]
fn negative_test_bitwise_operation_lookup_add_wrong_z() {
    // 5 + 7 = 12
    run_negative_test((5, 7, 11, BitwiseOperationLookupOpcode::ADD));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_add_x_out_of_range() {
    // (16 + 1) % 16 = 1, but need x < 2^NUM_BITS
    run_negative_test((16, 1, 1, BitwiseOperationLookupOpcode::ADD));
}

#[test]
fn negative_test_bitwise_operation_lookup_add_y_out_of_range() {
    // (1 + 16) % 16 = 1, but need y < 2^NUM_BITS
    run_negative_test((1, 16, 1, BitwiseOperationLookupOpcode::ADD));
}

#[test]
fn negative_test_bitwise_operation_lookup_add_no_mod() {
    // (8 + 8) % 16 = 0
    run_negative_test((8, 8, 16, BitwiseOperationLookupOpcode::ADD));
}

#[test]
fn negative_test_bitwise_operation_lookup_add_wrong_op() {
    // 5 + 7 = 12, 0101(5)) ^ 0111(7) = 0010(2)
    run_negative_test((5, 7, 12, BitwiseOperationLookupOpcode::XOR));
}

#[test]
fn negative_test_bitwise_operation_lookup_xor_wrong_z() {
    // 1011(11) ^ 0101(5) = 1110(14)
    run_negative_test((11, 5, 15, BitwiseOperationLookupOpcode::XOR));
}

#[test]
#[should_panic]
fn negative_test_bitwise_operation_lookup_xor_x_out_of_range() {
    // 10000(16) ^ 0001(1) = 0001(1) in 4 bits, but need x < 2^NUM_BITS
    run_negative_test((16, 1, 1, BitwiseOperationLookupOpcode::XOR));
}

#[test]
fn negative_test_bitwise_operation_lookup_xor_y_out_of_range() {
    // 0001(1) ^ 10000(16) = 0001(1) in 4 bits, but need y < 2^NUM_BITS
    run_negative_test((1, 16, 1, BitwiseOperationLookupOpcode::XOR));
}

#[test]
fn negative_test_bitwise_operation_lookup_xor_wrong_op() {
    // 5 + 7 = 12, 0101(5)) ^ 0111(7) = 0010(2)
    run_negative_test((5, 7, 2, BitwiseOperationLookupOpcode::ADD));
}
