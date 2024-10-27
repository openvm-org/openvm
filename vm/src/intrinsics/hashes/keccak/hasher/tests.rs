use std::{borrow::BorrowMut, sync::Arc};

use ax_circuit_primitives::xor::XorLookupChip;
use ax_stark_backend::{utils::disable_debug_builder, verifier::VerificationError};
use ax_stark_sdk::{config::baby_bear_blake3::BabyBearBlake3Config, utils::create_seeded_rng};
use axvm_instructions::instruction::Instruction;
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use p3_keccak_air::NUM_ROUNDS;
use rand::Rng;
use tiny_keccak::Hasher;

use super::{utils::num_keccak_f, KeccakVmChip};
use crate::{
    arch::{
        instructions::Keccak256Opcode,
        testing::{VmChipTestBuilder, VmChipTester},
        BYTE_XOR_BUS,
    },
    intrinsics::hashes::keccak::hasher::columns::KeccakVmCols,
};

// io is vector of (input, prank_output) where prank_output is Some if the trace
// will be replaced
fn build_keccak256_test(
    io: Vec<(Vec<u8>, Option<[u8; 32]>)>,
) -> VmChipTester<BabyBearBlake3Config> {
    let mut tester = VmChipTestBuilder::default();
    let xor_chip = Arc::new(XorLookupChip::<8>::new(BYTE_XOR_BUS));
    let mut chip = KeccakVmChip::new(
        tester.execution_bus(),
        tester.program_bus(),
        tester.memory_controller(),
        xor_chip.clone(),
        0,
    );

    let mut dst = 0;
    let src = 0;

    for (input, prank_output) in &io {
        let [a, b, c] = [0, 1, 2];
        let [d, e, f] = [1, 2, 1];

        tester.write_cell(d, a, BabyBear::from_canonical_usize(dst));
        tester.write_cell(d, b, BabyBear::from_canonical_usize(src));
        tester.write_cell(f, c, BabyBear::from_canonical_usize(input.len()));
        for (i, byte) in input.iter().enumerate() {
            tester.write_cell(e, src + i, BabyBear::from_canonical_usize(*byte as usize));
        }

        tester.execute(
            &mut chip,
            Instruction::large_from_isize(
                Keccak256Opcode::KECCAK256 as usize,
                a as isize,
                b as isize,
                c as isize,
                d as isize,
                e as isize,
                f as isize,
                0,
            ),
        );
        if let Some(output) = prank_output {
            for i in 0..16 {
                chip.records.last_mut().unwrap().digest_writes[i / 8].data[i % 8] =
                    BabyBear::from_canonical_u16(
                        output[2 * i] as u16 + ((output[2 * i + 1] as u16) << 8),
                    );
            }
        }
        // shift dst to not deal with timestamps for pranking
        dst += 16;
    }
    let mut tester = tester.build().load(chip).load(xor_chip).finalize();

    let keccak_trace = tester.air_proof_inputs[2].raw.common_main.as_mut().unwrap();
    let mut row = 0;
    for (input, output) in io {
        let num_blocks = num_keccak_f(input.len());
        let num_rows = NUM_ROUNDS * num_blocks;
        row += num_rows;
        if output.is_none() {
            continue;
        }
        let output = output.unwrap();
        let digest_row: &mut KeccakVmCols<_> = keccak_trace.row_mut(row - 1).borrow_mut();
        for i in 0..16 {
            let out_limb = BabyBear::from_canonical_u16(
                output[2 * i] as u16 + ((output[2 * i + 1] as u16) << 8),
            );
            let x = i / 4;
            let y = 0;
            let limb = i % 4;
            if x == 0 && y == 0 {
                digest_row.inner.a_prime_prime_prime_0_0_limbs[limb] = out_limb;
            } else {
                digest_row.inner.a_prime_prime[y][x][limb] = out_limb;
            }
        }
    }

    tester
}

#[test]
fn negative_test_keccak256() {
    let mut rng = create_seeded_rng();
    let mut hasher = tiny_keccak::Keccak::v256();
    let input: Vec<_> = vec![0; 137];
    hasher.update(&input);
    let mut out = [0u8; 32];
    hasher.finalize(&mut out);
    out[0] = rng.gen();
    let tester = build_keccak256_test(vec![(input, Some(out))]);
    disable_debug_builder();
    assert_eq!(
        tester.simple_test().err(),
        Some(VerificationError::OodEvaluationMismatch)
    );
}
