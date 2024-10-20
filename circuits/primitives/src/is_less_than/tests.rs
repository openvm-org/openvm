use std::sync::Arc;

use afs_stark_backend::{
    interaction::InteractionBuilder,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use ax_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
use derive_new::new;
use p3_air::{Air, BaseAir};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, Field};
use p3_matrix::{
    dense::{DenseMatrix, RowMajorMatrix},
    Matrix,
};

use super::IsLessThanIo;
use crate::{
    is_less_than::IsLessThanAir,
    var_range::{VariableRangeCheckerBus, VariableRangeCheckerChip},
    SubAir, TraceSubRowGenerator,
};

/// Struct purely for testing purposes. We could make this have a const generic just like
/// `AssertLessThanCols`, but for demonstration purposes we use `Vec` to show how to use the
/// SubAir even when the columns do not implement `AlignedBorrow`.
#[derive(Clone, Debug, new)]
pub struct IsLessThanCols<T> {
    pub x: T,
    pub y: T,
    pub out: T,
    pub lower_decomp: Vec<T>,
}

/// Note that this air has no const generics. The parameters such as `max_bits, decomp_limbs` are all
/// configured in the constructor at runtime.
#[derive(Clone, Copy)]
pub struct IsLtTestAir(pub IsLessThanAir);

impl<F: Field> BaseAirWithPublicValues<F> for IsLtTestAir {}
impl<F: Field> PartitionedBaseAir<F> for IsLtTestAir {}
impl<F: Field> BaseAir<F> for IsLtTestAir {
    fn width(&self) -> usize {
        // Cannot use size_of because Cols has Vec<T> which is stored on the heap
        3 + self.0.decomp_limbs
    }
}
impl<AB: InteractionBuilder> Air<AB> for IsLtTestAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let (io, lower_decomp) = local.split_at(3);
        let [x, y, out] = [io[0], io[1], io[2]];

        let io = IsLessThanIo::new(x, y, out, AB::F::one());
        self.0.eval(builder, (io, lower_decomp));
    }
}

#[derive(Clone)]
pub struct IsLessThanChip {
    pub air: IsLtTestAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
}

impl IsLessThanChip {
    pub fn new(max_bits: usize, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let bus = range_checker.bus();
        Self {
            air: IsLtTestAir(IsLessThanAir::new(bus, max_bits)),
            range_checker,
        }
    }
    pub fn generate_trace<F: Field>(&self, pairs: Vec<(u32, u32)>) -> RowMajorMatrix<F> {
        assert!(pairs.len().is_power_of_two());
        let width: usize = BaseAir::<F>::width(&self.air);

        let mut rows = vec![F::zero(); width * pairs.len()];
        for (row, (x, y)) in rows.chunks_mut(width).zip(pairs) {
            let mut row = IsLessThanColsMut::from_mut_slice(row);
            *row.x = F::from_canonical_u32(x);
            *row.y = F::from_canonical_u32(y);
            *row.out = F::from_bool(x < y);
            self.air
                .0
                .generate_subrow((&self.range_checker, x, y), &mut row.lower_decomp);
        }

        RowMajorMatrix::new(rows, width)
    }
}

// We create a custom struct of mutable references since `IsLessThanCols` cannot derive `AlignedBorrow`.
pub struct IsLessThanColsMut<'a, T> {
    pub x: &'a mut T,
    pub y: &'a mut T,
    pub out: &'a mut T,
    pub lower_decomp: &'a mut [T],
}

impl<'a, T> IsLessThanColsMut<'a, T> {
    pub fn from_mut_slice(slc: &'a mut [T]) -> Self {
        let (io, lower_decomp) = slc.split_at_mut(3);
        let mut io = io.iter_mut();

        Self {
            x: io.next().unwrap(),
            y: io.next().unwrap(),
            out: io.next().unwrap(),
            lower_decomp,
        }
    }
}

fn get_tester_is_lt_chip() -> IsLessThanChip {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);

    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    IsLessThanChip::new(max_bits, range_checker)
}

#[test]
fn test_is_less_than_chip_lt() {
    let chip = get_tester_is_lt_chip();
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let chip = get_tester_is_lt_chip();
    let trace = chip.generate_trace(vec![(14321, 26883), (1, 0), (773, 773), (337, 456)]);
    let range_trace: DenseMatrix<BabyBear> = chip.range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![chip.air, chip.range_checker.air],
        vec![trace, range_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_negative() {
    let chip = get_tester_is_lt_chip();
    let mut trace = chip.generate_trace(vec![(446, 553)]);
    let range_trace = chip.range_checker.generate_trace();

    trace.values[2] = AbstractField::from_canonical_u64(0);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![chip.air, chip.range_checker.air],
            vec![trace, range_trace],
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}
