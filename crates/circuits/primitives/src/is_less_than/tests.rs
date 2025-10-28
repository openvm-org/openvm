use std::sync::Arc;

use derive_new::new;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
    utils::disable_debug_builder,
    verifier::VerificationError,
};
use openvm_stark_sdk::{
    any_rap_arc_vec, config::baby_bear_poseidon2::BabyBearPoseidon2Engine, engine::StarkFriEngine,
};
#[cfg(feature = "cuda")]
use {
    crate::cuda_abi::less_than::less_than_dummy_tracegen,
    openvm_cuda_backend::{
        base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix, types::F,
    },
    openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer},
};

use super::IsLessThanIo;
use crate::{
    is_less_than::IsLtSubAir,
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

/// Note that this air has no const generics. The parameters such as `max_bits, decomp_limbs` are
/// all configured in the constructor at runtime.
#[derive(Clone, Copy)]
pub struct IsLtTestAir(pub IsLtSubAir);

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

        let io = IsLessThanIo::new(x, y, out, AB::F::ONE);
        self.0.eval(builder, (io, lower_decomp));
    }
}

pub struct IsLessThanChip {
    pub air: IsLtTestAir,
    pub range_checker: Arc<VariableRangeCheckerChip>,
    pub pairs: Vec<(u32, u32)>,
}

impl IsLessThanChip {
    pub fn new(max_bits: usize, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let bus = range_checker.bus();
        Self {
            air: IsLtTestAir(IsLtSubAir::new(bus, max_bits)),
            range_checker,
            pairs: vec![],
        }
    }
    pub fn generate_trace<F: PrimeField32>(self) -> RowMajorMatrix<F> {
        assert!(self.pairs.len().is_power_of_two());
        let width: usize = BaseAir::<F>::width(&self.air);

        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row = IsLessThanColsMut::from_mut_slice(row);
                *row.x = F::from_canonical_u32(x);
                *row.y = F::from_canonical_u32(y);
                self.air
                    .0
                    .generate_subrow((&self.range_checker, x, y), (row.lower_decomp, row.out));
            });

        RowMajorMatrix::new(rows, width)
    }
}

// We create a custom struct of mutable references since `IsLessThanCols` cannot derive
// `AlignedBorrow`.
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

fn setup() -> (IsLessThanChip, Arc<VariableRangeCheckerChip>) {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    let bus = VariableRangeCheckerBus::new(0, decomp);
    let range_checker = Arc::new(VariableRangeCheckerChip::new(bus));

    (
        IsLessThanChip::new(max_bits, range_checker.clone()),
        range_checker,
    )
}

#[test]
fn test_is_less_than_chip_lt() {
    let (mut chip, range_checker) = setup();
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(14321, 26883), (1, 0), (773, 773), (337, 456)];
    let trace = chip.generate_trace();
    let range_trace = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_lt_chip_decomp_does_not_divide() {
    let (mut chip, range_checker) = setup();
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(14321, 26883), (1, 0), (773, 773), (337, 456)];
    let trace = chip.generate_trace();
    let range_trace = range_checker.generate_trace();

    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace])
        .expect("Verification failed");
}

#[test]
fn test_is_less_than_negative() {
    let (mut chip, range_checker) = setup();
    let airs = any_rap_arc_vec![chip.air, range_checker.air];
    chip.pairs = vec![(446, 553)];
    let mut trace = chip.generate_trace();
    let range_trace = range_checker.generate_trace();

    trace.values[2] = FieldAlgebra::from_canonical_u64(0);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(airs, vec![trace, range_trace],).err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_less_than_tracegen() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    const AUX_LEN: usize = 2;

    let num_pairs = 4;
    let trace = DeviceMatrix::<F>::with_capacity(num_pairs, 3 + AUX_LEN);
    let pairs = vec![[14321, 26883], [1, 0], [773, 773], [337, 456]]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .to_device()
        .unwrap();

    let rc_num_bins = (1 << (decomp + 1)) as usize;
    let rc_histogram = DeviceBuffer::<u32>::with_capacity(rc_num_bins);

    unsafe {
        less_than_dummy_tracegen(
            trace.buffer(),
            num_pairs,
            &pairs,
            max_bits,
            AUX_LEN,
            &rc_histogram,
        )
        .unwrap();
    }

    // From test_lt_chip_decomp_does_not_divide
    let expected_cpu_matrix_vals: [[u32; 5]; 4] = [
        [14321, 26883, 1, 17, 49],
        [1, 0, 0, 254, 255],
        [773, 773, 0, 255, 255],
        [337, 456, 1, 118, 0],
    ];
    let expected_cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        expected_cpu_matrix_vals
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect(),
        3 + AUX_LEN,
    ));

    assert_eq_host_and_device_matrix(expected_cpu_matrix, &trace);
}
