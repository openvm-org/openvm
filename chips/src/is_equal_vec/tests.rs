use afs_stark_backend::{prover::USE_DEBUG_BUILDER, verifier::VerificationError};
use afs_test_utils::{
    config::baby_bear_poseidon2::run_simple_test_no_pis, utils::create_seeded_rng,
};
use p3_baby_bear::BabyBear;
use p3_field::AbstractField;
use rand::Rng;

use crate::is_equal_vec::IsEqualVecChip;

#[test]
fn test_single_is_equal_vec() {
    let x = vec![1, 2, 3]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = vec![1, 2, 3]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec2() {
    let x = vec![2, 2, 7]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = vec![3, 5, 1]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec3() {
    let x = vec![17, 23, 4]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = vec![17, 23, 4]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let trace = chip.generate_trace(vec![x], vec![y]);

    run_simple_test_no_pis(vec![&chip], vec![trace]).expect("Verification failed");
}

#[test]
fn test_single_is_equal_vec4() {
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

#[test]
fn test_single_is_equal_vec_fail() {
    let x = vec![1, 2, 3]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = vec![1, 2, 1]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let mut trace = chip.generate_trace(vec![x], vec![y]);

    trace.values[0] = AbstractField::from_canonical_u32(2);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_single_is_equal_vec_fail2() {
    let x = vec![1, 2, 3]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();
    let y = vec![1, 2, 1]
        .into_iter()
        .map(AbstractField::from_canonical_u32)
        .collect();

    let chip = IsEqualVecChip { vec_len: 3 };

    let mut trace = chip.generate_trace(vec![x], vec![y]);

    trace.values[8] = AbstractField::from_canonical_u32(1);

    USE_DEBUG_BUILDER.with(|debug| {
        *debug.lock().unwrap() = false;
    });
    assert_eq!(
        run_simple_test_no_pis(vec![&chip], vec![trace]),
        Err(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_all_is_equal_vec_fail() {
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
