use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::is_equal_vec::IsEqualVecChip;

use test_case::test_case;

#[test_case([1, 2, 3], [1, 2, 3], [1, 1, 1] ; "1, 2, 3 == 1, 2, 3")]
#[test_case([1, 2, 3], [1, 2, 1], [1, 1, 0] ; "1, 2, 3 != 1, 2, 1")]
#[test_case([2, 2, 7], [3, 5, 1], [0, 0, 0] ; "2, 2, 7 != 3, 5, 1")]
#[test_case([17, 23, 4], [17, 23, 4], [1, 1, 1] ; "17, 23, 4 == 17, 23, 4")]
#[test_case([92, 27, 32], [92, 27, 32], [1, 1, 1] ; "92, 27, 32 == 92, 27, 32")]
#[test_case([1, 27, 4], [1, 2, 43], [1, 0, 0] ; "1, 27, 4 != 1, 2, 43")]
fn test_vec_is_equal_vec(x: [u32; 3], y: [u32; 3], expected: [u32; 3]) {
    let x = x
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = y
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let trace = chip.generate_trace(vec![x], vec![y]);

    for (i, value) in expected.iter().enumerate() {
        assert_eq!(
            trace.values[6 + i],
            AbstractField::from_canonical_u32(*value)
        );
    }

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_all_is_equal_vec() {
    let x1 = vec![1, 2, 3];
    let y1 = vec![1, 2, 1];
    let x2 = vec![2, 2, 7];
    let y2 = vec![3, 5, 1];
    let x3 = vec![17, 23, 4];
    let y3 = vec![17, 23, 4];
    let x4 = vec![1, 2, 3];
    let y4 = vec![1, 2, 1];

    let chip = IsEqualVecChip { vec_len: 3 };

    let trace = chip.generate_trace(
        vec![x1, x2, x3, x4]
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(AbstractField::from_canonical_u32)
                    .collect()
            })
            .collect(),
        vec![y1, y2, y3, y4]
            .into_iter()
            .map(|v| {
                v.into_iter()
                    .map(AbstractField::from_canonical_u32)
                    .collect()
            })
            .collect(),
    );

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test_case([1, 2, 3], [1, 2, 3], [1, 1, 1] ; "1, 2, 3 == 1, 2, 3")]
#[test_case([1, 2, 3], [1, 2, 1], [1, 1, 0] ; "1, 2, 3 != 1, 2, 1")]
#[test_case([2, 2, 7], [3, 5, 1], [0, 0, 0] ; "2, 2, 7 != 3, 5, 1")]
#[test_case([17, 23, 4], [17, 23, 4], [1, 1, 1] ; "17, 23, 4 == 17, 23, 4")]
#[test_case([92, 27, 32], [92, 27, 32], [1, 1, 1] ; "92, 27, 32 == 92, 27, 32")]
#[test_case([1, 27, 4], [1, 2, 43], [1, 0, 0] ; "1, 27, 4 != 1, 2, 43")]
fn test_single_is_equal_vec_fail(x: [u32; 3], y: [u32; 3], expected: [u32; 3]) {
    let x: Vec<BabyBear> = x
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y: Vec<BabyBear> = y
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let mut trace = chip.generate_trace(vec![x], vec![y]);

    for (i, _value) in expected.iter().enumerate() {
        trace.values[6 + i] = BabyBear::one() - trace.values[6 + i];
        USE_DEBUG_BUILDER.with(|debug| {
            *debug.lock().unwrap() = false;
        });
        assert_eq!(
            run_simple_test_no_pis(vec![&chip], vec![trace.clone()]),
            Err(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        trace.values[6 + i] = BabyBear::one() - trace.values[6 + i];
    }
}

#[test]
fn test_vec_is_equal_vec_fail() {
    let width = 4;
    let height = 2;
    let mut rng = create_seeded_rng();
    let x: Vec<Vec<BabyBear>> = (0..height)
        .map(|_| {
            (0..width)
                .map(|_| AbstractField::from_wrapped_u32(rng.gen::<u32>()))
                .collect()
        })
        .collect();
    let y: Vec<Vec<_>> = x.clone();
    let chip = IsEqualVecChip { vec_len: width };

    for i in 0..height {
        for j in 0..width {
            let mut trace = chip.generate_trace(x.clone(), y.clone());
            trace.values[i * width + j] += AbstractField::from_wrapped_u32(rng.gen::<u32>() + 1);
            USE_DEBUG_BUILDER.with(|debug| {
                *debug.lock().unwrap() = false;
            });
            assert_eq!(
                run_simple_test_no_pis(vec![&chip], vec![trace.clone()]),
                Err(VerificationError::OodEvaluationMismatch),
                "Expected constraint to fail"
            );
            trace.row_mut(i)[j] = BabyBear::one() - trace.row_mut(i)[j];
            USE_DEBUG_BUILDER.with(|debug| {
                *debug.lock().unwrap() = false;
            });
            assert_eq!(
                run_simple_test_no_pis(vec![&chip], vec![trace.clone()]),
                Err(VerificationError::OodEvaluationMismatch),
                "Expected constraint to fail"
            );
        }
    }
}
