use std::borrow::{Borrow, BorrowMut};

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
    p3_baby_bear::BabyBear,
};
use test_case::test_case;

use super::{IsZeroIo, IsZeroSubAir};
use crate::{SubAir, TraceSubRowGenerator};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsZeroCols<T> {
    pub x: T,
    pub out: T,
    pub inv: T,
}

#[derive(Copy, Clone)]
pub struct IsZeroTestAir(IsZeroSubAir);

impl<F: Field> BaseAirWithPublicValues<F> for IsZeroTestAir {}
impl<F: Field> PartitionedBaseAir<F> for IsZeroTestAir {}
impl<F: Field> BaseAir<F> for IsZeroTestAir {
    fn width(&self) -> usize {
        IsZeroCols::<F>::width()
    }
}
impl<AB: AirBuilder> Air<AB> for IsZeroTestAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &IsZeroCols<_> = (*local).borrow();
        let io = IsZeroIo::new(local.x.into(), local.out.into(), AB::Expr::ONE);

        self.0.eval(builder, (io, local.inv));
    }
}

pub struct IsZeroChip<F> {
    air: IsZeroTestAir,
    x: Vec<F>,
}

impl<F: Field> IsZeroChip<F> {
    pub fn new(x: Vec<F>) -> Self {
        Self {
            air: IsZeroTestAir(IsZeroSubAir),
            x,
        }
    }

    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        let air = IsZeroSubAir;
        assert!(self.x.len().is_power_of_two());
        let width = IsZeroCols::<F>::width();
        let mut rows = F::zero_vec(width * self.x.len());
        rows.par_chunks_mut(width).zip(self.x).for_each(|(row, x)| {
            let row: &mut IsZeroCols<F> = row.borrow_mut();
            row.x = x;
            air.generate_subrow(x, (&mut row.inv, &mut row.out));
        });

        RowMajorMatrix::new(rows, width)
    }
}

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero(x: u32) {
    let chip = IsZeroChip::new(vec![BabyBear::from_canonical_u32(x)]);
    let air = chip.air;
    let trace = chip.generate_trace();

    assert_eq!(trace.get(0, 1), FieldAlgebra::from_bool(x == 0));

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
        .expect("Verification failed");
}

#[test_case([0, 1, 2, 7], [1, 0, 0, 0] ; "0, 1, 2, 7 => 1, 0, 0, 0")]
#[test_case([97, 23, 179, 0], [0, 0, 0, 1] ; "97, 23, 179, 0 => 0, 0, 0, 1")]
fn test_vec_is_zero(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec = x_vec
        .into_iter()
        .map(FieldAlgebra::from_canonical_u32)
        .collect();
    let chip = IsZeroChip::new(x_vec);
    let air = chip.air;
    let trace = chip.generate_trace();

    for (i, value) in expected.iter().enumerate() {
        assert_eq!(
            trace.values[3 * i + 1],
            FieldAlgebra::from_canonical_u32(*value)
        );
    }

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
        .expect("Verification failed");
}

#[test_case(97 ; "97 => 0")]
#[test_case(0 ; "0 => 1")]
fn test_single_is_zero_fail(x: u32) {
    let x = FieldAlgebra::from_canonical_u32(x);
    let chip = IsZeroChip::new(vec![x]);
    let air = chip.air;
    let mut trace = chip.generate_trace();
    trace.values[1] = BabyBear::ONE - trace.values[1];

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(any_rap_arc_vec![air], vec![trace])
            .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected constraint to fail"
    );
}

#[test_case([1, 2, 7, 0], [0, 0, 0, 1] ; "1, 2, 7, 0 => 0, 0, 0, 1")]
#[test_case([97, 0, 179, 0], [0, 1, 0, 1] ; "97, 0, 179, 0 => 0, 1, 0, 1")]
fn test_vec_is_zero_fail(x_vec: [u32; 4], expected: [u32; 4]) {
    let x_vec: Vec<BabyBear> = x_vec
        .into_iter()
        .map(BabyBear::from_canonical_u32)
        .collect();
    let chip = IsZeroChip::new(x_vec);
    let air = chip.air;
    let mut trace = chip.generate_trace();

    disable_debug_builder();
    for (i, _value) in expected.iter().enumerate() {
        trace.row_mut(i)[1] = BabyBear::ONE - trace.row_mut(i)[1];
        assert_eq!(
            BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
                any_rap_arc_vec![air],
                vec![trace.clone()]
            )
            .err(),
            Some(VerificationError::OodEvaluationMismatch),
            "Expected constraint to fail"
        );
        trace.row_mut(i)[1] = BabyBear::ONE - trace.row_mut(i)[1];
    }
}
