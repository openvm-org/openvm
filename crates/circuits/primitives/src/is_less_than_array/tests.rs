use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    p3_air::{Air, BaseAir},
    p3_field::{Field, FieldAlgebra},
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
    crate::cuda_abi::less_than::less_than_array_dummy_tracegen,
    openvm_cuda_backend::{
        base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix, types::F,
    },
    openvm_cuda_common::{copy::MemCopyH2D as _, d_buffer::DeviceBuffer},
};

use super::*;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Copy, Debug)]
pub struct IsLtArrayCols<T, const NUM: usize, const AUX_LEN: usize> {
    pub x: [T; NUM],
    pub y: [T; NUM],
    pub out: T,
    pub aux: IsLtArrayAuxCols<T, NUM, AUX_LEN>,
}

#[derive(Clone, Copy)]
pub struct IsLtArrayTestAir<const NUM: usize, const AUX_LEN: usize>(IsLtArraySubAir<NUM>);

impl<F: Field, const NUM: usize, const AUX_LEN: usize> BaseAirWithPublicValues<F>
    for IsLtArrayTestAir<NUM, AUX_LEN>
{
}
impl<F: Field, const NUM: usize, const AUX_LEN: usize> BaseAir<F>
    for IsLtArrayTestAir<NUM, AUX_LEN>
{
    fn width(&self) -> usize {
        IsLtArrayCols::<F, NUM, AUX_LEN>::width()
    }
}
impl<F: Field, const NUM: usize, const AUX_LEN: usize> PartitionedBaseAir<F>
    for IsLtArrayTestAir<NUM, AUX_LEN>
{
}

impl<AB: InteractionBuilder, const NUM: usize, const AUX_LEN: usize> Air<AB>
    for IsLtArrayTestAir<NUM, AUX_LEN>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &IsLtArrayCols<AB::Var, NUM, AUX_LEN> = (*local).borrow();

        let io = IsLtArrayIo {
            x: local.x.map(Into::into),
            y: local.y.map(Into::into),
            out: local.out.into(),
            count: AB::Expr::ONE,
        };
        self.0.eval(builder, (io, (&local.aux).into()));
    }
}

/// This chip computes whether one tuple is lexicographically less than another. Each element of the
/// tuple has its own max number of bits, given by the limb_bits array. The chip assumes that each
/// limb is within its given max limb_bits.
///
/// The IsLessThanTupleChip uses the IsLessThanChip as a subchip to check whether individual tuple
/// elements are less than each other.
pub struct IsLtArrayChip<const NUM: usize, const AUX_LEN: usize> {
    pub air: IsLtArrayTestAir<NUM, AUX_LEN>,
    pub range_checker: Arc<VariableRangeCheckerChip>,

    pub pairs: Vec<([u32; NUM], [u32; NUM])>,
}

impl<const NUM: usize, const AUX_LEN: usize> IsLtArrayChip<NUM, AUX_LEN> {
    pub fn new(max_bits: usize, range_checker: Arc<VariableRangeCheckerChip>) -> Self {
        let air = IsLtArrayTestAir(IsLtArraySubAir::new(range_checker.bus(), max_bits));
        Self {
            air,
            range_checker,
            pairs: vec![],
        }
    }

    pub fn generate_trace<F: PrimeField32>(self) -> RowMajorMatrix<F> {
        assert!(self.pairs.len().is_power_of_two());
        let width = BaseAir::<F>::width(&self.air);
        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut IsLtArrayCols<_, NUM, AUX_LEN> = row.borrow_mut();
                row.x = x.map(F::from_canonical_u32);
                row.y = y.map(F::from_canonical_u32);
                self.air.0.generate_subrow(
                    (&self.range_checker, &row.x, &row.y),
                    ((&mut row.aux).into(), &mut row.out),
                );
            });
        RowMajorMatrix::new(rows, width)
    }

    pub fn generate_wrong_trace<F: PrimeField32>(self) -> RowMajorMatrix<F> {
        assert!(self.pairs.len().is_power_of_two());
        let width = BaseAir::<F>::width(&self.air);
        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut IsLtArrayCols<_, NUM, AUX_LEN> = row.borrow_mut();
                row.x = x.map(F::from_canonical_u32);
                row.y = y.map(F::from_canonical_u32);
                row.out = F::ZERO;
                let aux: IsLtArrayAuxColsMut<_> = (&mut row.aux).into();
                aux.diff_marker
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, diff_marker)| {
                        *diff_marker = if i == 0 { F::ONE } else { F::ZERO };
                    });
                *aux.diff_inv = F::ZERO;
                self.range_checker.decompose(
                    (1 << self.air.0.max_bits()) - 1,
                    self.air.0.max_bits(),
                    aux.lt_decomp,
                );
            });
        RowMajorMatrix::new(rows, width)
    }
}

fn get_range_bus() -> VariableRangeCheckerBus {
    let range_max_bits: usize = 8;
    VariableRangeCheckerBus::new(0, range_max_bits)
}
fn get_tester_range_chip() -> Arc<VariableRangeCheckerChip> {
    let bus = get_range_bus();
    Arc::new(VariableRangeCheckerChip::new(bus))
}

const N: usize = 2;
const LIMBS: usize = 2;

#[test]
fn test_is_less_than_tuple_chip() {
    let range_checker = get_tester_range_chip();
    let mut chip = IsLtArrayChip::<N, LIMBS>::new(16, range_checker.clone());
    let air = chip.air;
    chip.pairs = vec![
        ([14321, 123], [26678, 233]),
        ([26678, 244], [14321, 233]),
        ([14321, 244], [14321, 244]),
        ([26678, 233], [14321, 244]),
    ];

    let trace = chip.generate_trace();
    let range_checker_trace = range_checker.generate_trace();
    BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
        any_rap_arc_vec![air, range_checker.air],
        vec![trace, range_checker_trace],
    )
    .expect("Verification failed");
}

#[test]
fn test_is_less_than_tuple_chip_negative() {
    let range_checker = get_tester_range_chip();
    let mut chip = IsLtArrayChip::<N, LIMBS>::new(16, range_checker.clone());
    let air = chip.air;
    chip.pairs = vec![([14321, 123], [26678, 233])];
    let mut trace = chip.generate_trace();
    let range_checker_trace = range_checker.generate_trace();

    trace.values[2] = FieldAlgebra::from_canonical_u64(0);

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_checker_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[test]
fn test_is_less_than_tuple_chip_nonzero_diff() {
    let range_checker = get_tester_range_chip();
    let mut chip = IsLtArrayChip::<N, LIMBS>::new(16, range_checker.clone());
    let air = chip.air;
    chip.pairs = vec![([0, 0], [0, 1])];

    let trace = chip.generate_wrong_trace();
    let range_checker_trace = range_checker.generate_trace();

    disable_debug_builder();
    assert_eq!(
        BabyBearPoseidon2Engine::run_simple_test_no_pis_fast(
            any_rap_arc_vec![air, range_checker.air],
            vec![trace, range_checker_trace]
        )
        .err(),
        Some(VerificationError::OodEvaluationMismatch),
        "Expected verification to fail, but it passed"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_less_than_array_tracegen() {
    let max_bits: usize = 16;
    let decomp: usize = 8;
    const ARRAY_LEN: usize = 2;
    const AUX_LEN: usize = 2;

    let num_pairs = 4;
    let trace = DeviceMatrix::<F>::with_capacity(num_pairs, 3 * ARRAY_LEN + AUX_LEN + 2);
    let pairs = vec![
        [14321, 123, 26678, 233],
        [26678, 244, 14321, 233],
        [14321, 244, 14321, 244],
        [14321, 233, 14321, 244],
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .to_device()
    .unwrap();

    let rc_num_bins = (1 << (decomp + 1)) as usize;
    let rc_histogram = DeviceBuffer::<u32>::with_capacity(rc_num_bins);

    unsafe {
        less_than_array_dummy_tracegen(
            trace.buffer(),
            num_pairs,
            &pairs,
            max_bits,
            ARRAY_LEN,
            AUX_LEN,
            &rc_histogram,
        )
        .unwrap();
    }

    // From test_is_less_than_tuple_chip in OpenVM's is_less_than_array tests, modified slightly
    let expected_cpu_matrix_vals = [
        [14321, 123, 26678, 233, 1, 1, 0, 1344947008, 68, 48],
        [26678, 244, 14321, 233, 0, 1, 0, 668318913, 186, 207],
        [14321, 244, 14321, 244, 0, 0, 0, 0, 255, 255],
        [14321, 233, 14321, 244, 1, 0, 1, 549072524, 10, 0],
    ];
    let expected_cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
        expected_cpu_matrix_vals
            .into_iter()
            .flatten()
            .map(F::from_canonical_u32)
            .collect(),
        3 * ARRAY_LEN + AUX_LEN + 2,
    ));

    assert_eq_host_and_device_matrix(expected_cpu_matrix, &trace);
}
