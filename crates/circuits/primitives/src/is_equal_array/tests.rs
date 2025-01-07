use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
    p3_baby_bear::BabyBear, utils::create_seeded_rng,
};
use rand::Rng;
use test_case::test_case;

use super::{IsEqArrayAuxCols, IsEqArrayIo, IsEqArraySubAir};
use crate::{SubAir, TraceSubRowGenerator};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqArrayCols<T, const N: usize> {
    x: [T; N],
    y: [T; N],
    out: T,
    aux: IsEqArrayAuxCols<T, N>,
}

#[derive(Clone, Copy)]
pub struct IsEqArrayTestAir<const N: usize>(IsEqArraySubAir<N>);

impl<F: Field, const N: usize> BaseAirWithPublicValues<F> for IsEqArrayTestAir<N> {}
impl<F: Field, const N: usize> PartitionedBaseAir<F> for IsEqArrayTestAir<N> {}
impl<F: Field, const N: usize> BaseAir<F> for IsEqArrayTestAir<N> {
    fn width(&self) -> usize {
        IsEqArrayCols::<F, N>::width()
    }
}
impl<AB: AirBuilder, const N: usize> Air<AB> for IsEqArrayTestAir<N> {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &IsEqArrayCols<AB::Var, N> = (*local).borrow();
        let io = IsEqArrayIo {
            x: local.x.map(Into::into),
            y: local.y.map(Into::into),
            out: local.out.into(),
            condition: AB::Expr::ONE,
        };
        self.0.eval(builder, (io, local.aux.diff_inv_marker));
    }
}

pub struct IsEqArrayChip<F, const N: usize> {
    air: IsEqArrayTestAir<N>,
    pairs: Vec<([F; N], [F; N])>,
}

impl<F: Field, const N: usize> IsEqArrayChip<F, N> {
    pub fn new(pairs: Vec<([F; N], [F; N])>) -> Self {
        let air = IsEqArrayTestAir(IsEqArraySubAir);
        Self { air, pairs }
    }
    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        let air: IsEqArraySubAir<N> = IsEqArraySubAir;
        assert!(self.pairs.len().is_power_of_two());
        let width = IsEqArrayCols::<F, N>::width();
        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut IsEqArrayCols<F, N> = row.borrow_mut();
                air.generate_subrow((&x, &y), (&mut row.aux.diff_inv_marker, &mut row.out));
                row.x = x;
                row.y = y;
            });

        RowMajorMatrix::new(rows, width)
    }
}

#[test_case([1, 2, 3], [1, 2, 3], 1; "1, 2, 3 == 1, 2, 3")]
#[test_case([1, 2, 3], [1, 2, 1], 0; "1, 2, 3 != 1, 2, 1")]
#[test_case([2, 2, 7], [3, 5, 1], 0; "2, 2, 7 != 3, 5, 1")]
#[test_case([17, 23, 4], [17, 23, 4], 1; "17, 23, 4 == 17, 23, 4")]
#[test_case([92, 27, 32], [92, 27, 32], 1; "92, 27, 32 == 92, 27, 32")]
#[test_case([1, 27, 4], [1, 2, 43], 0; "1, 27, 4 != 1, 2, 43")]
fn test_is_eq_array_single_row(x: [u32; 3], y: [u32; 3], is_equal: u32) {
    let x = x.map(FieldAlgebra::from_canonical_u32);
    let y = y.map(FieldAlgebra::from_canonical_u32);

    let chip = IsEqArrayChip::new(vec![(x, y)]);
    let air = chip.air;
    let trace = chip.generate_trace();
    let row: &IsEqArrayCols<BabyBear, 3> = trace.values.as_slice().borrow();

    assert_eq!(row.out, FieldAlgebra::from_canonical_u32(is_equal));

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
        .expect("Verification failed");
}

#[test]
fn test_is_eq_array_multi_rows() {
    let pairs = [
        ([1, 2, 3], [1, 2, 1]),
        ([2, 2, 7], [3, 5, 1]),
        ([17, 23, 4], [17, 23, 4]),
        ([1, 2, 3], [1, 2, 1]),
    ]
    .into_iter()
    .map(|(x, y)| {
        (
            x.map(FieldAlgebra::from_canonical_u32),
            y.map(FieldAlgebra::from_canonical_u32),
        )
    })
    .collect();

    let chip = IsEqArrayChip::new(pairs);
    let air = chip.air;

    let trace = chip.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
        .expect("Verification failed");
}

#[test_case([1, 2, 3], [1, 2, 3]; "1, 2, 3 == 1, 2, 3")]
#[test_case([1, 2, 3], [1, 2, 1]; "1, 2, 3 != 1, 2, 1")]
#[test_case([2, 2, 7], [3, 5, 1]; "2, 2, 7 != 3, 5, 1")]
#[test_case([17, 23, 4], [17, 23, 4]; "17, 23, 4 == 17, 23, 4")]
#[test_case([92, 27, 32], [92, 27, 32]; "92, 27, 32 == 92, 27, 32")]
#[test_case([1, 27, 4], [1, 2, 43]; "1, 27, 4 != 1, 2, 43")]
fn test_is_eq_array_single_row_fail(x: [u32; 3], y: [u32; 3]) {
    let x = x.map(FieldAlgebra::from_canonical_u32);
    let y = y.map(FieldAlgebra::from_canonical_u32);

    let chip = IsEqArrayChip::new(vec![(x, y)]);
    let air = chip.air;
    let mut trace = chip.generate_trace();

    disable_debug_builder();
    let row: &mut IsEqArrayCols<BabyBear, 3> = trace.values.as_mut_slice().borrow_mut();
    row.out = BabyBear::ONE - row.out;
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
            .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test]
fn test_is_eq_array_fail_rand() {
    const N: usize = 4;
    let height = 2;
    let mut rng = create_seeded_rng();
    let pairs: Vec<_> = (0..height)
        .map(|_| {
            let x = from_fn(|_| FieldAlgebra::from_wrapped_u32(rng.gen::<u32>()));
            (x, x)
        })
        .collect();
    let chip = IsEqArrayChip::<_, N>::new(pairs);
    let air = chip.air;
    let trace = chip.generate_trace();

    disable_debug_builder();
    for i in 0..height {
        for j in 0..N {
            let mut prank_trace = trace.clone();
            prank_trace.row_mut(i)[j] += FieldAlgebra::from_wrapped_u32(rng.gen::<u32>() + 1);
            assert_eq!(
                BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
                    any_rap_arc_vec![air],
                    vec![prank_trace]
                )
                .err(),
                Some(VerificationError::OodEvaluationMismatch),
                "Expected constraint to fail"
            );
        }
    }
}
