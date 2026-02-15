use std::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    any_air_arc_vec,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_maybe_rayon::prelude::*,
    prover::{AirProvingContext, ColMajorMatrix},
    utils::disable_debug_builder,
    BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine,
};
use test_case::test_matrix;
#[cfg(feature = "cuda")]
use {
    crate::cuda_abi::is_equal,
    openvm_cuda_backend::{
        base::DeviceMatrix, data_transporter::assert_eq_host_and_device_matrix, types::F,
    },
    openvm_cuda_common::copy::MemCopyH2D as _,
    openvm_stark_backend::p3_field::PrimeField32,
    openvm_stark_sdk::utils::create_seeded_rng,
    rand::Rng,
    std::sync::Arc,
};

use super::{IsEqSubAir, IsEqualIo};
use crate::{utils::test_engine_small, SubAir, TraceSubRowGenerator};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct IsEqualCols<T> {
    pub x: T,
    pub y: T,
    pub out: T,
    pub inv: T,
}

#[derive(Clone, Copy)]
pub struct IsEqTestAir(pub IsEqSubAir);

impl<F: Field> BaseAirWithPublicValues<F> for IsEqTestAir {}
impl<F: Field> PartitionedBaseAir<F> for IsEqTestAir {}
impl<F: Field> BaseAir<F> for IsEqTestAir {
    fn width(&self) -> usize {
        IsEqualCols::<F>::width()
    }
}
impl<AB> Air<AB> for IsEqTestAir
where
    AB: AirBuilder<F: Field, Var: Copy>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).expect("window should have two elements");
        let local: &IsEqualCols<_> = (*local).borrow();
        let io = IsEqualIo::new(
            local.x.into(),
            local.y.into(),
            local.out.into(),
            AB::Expr::ONE,
        );

        self.0.eval(builder, (io, local.inv));
    }
}

pub struct IsEqualChip<F> {
    pairs: Vec<(F, F)>,
}

impl<F: Field> IsEqualChip<F> {
    pub fn generate_trace(self) -> RowMajorMatrix<F> {
        let air = IsEqSubAir;
        assert!(self.pairs.len().is_power_of_two());
        let width = IsEqualCols::<F>::width();
        let mut rows = F::zero_vec(width * self.pairs.len());
        rows.par_chunks_mut(width)
            .zip(self.pairs)
            .for_each(|(row, (x, y))| {
                let row: &mut IsEqualCols<F> = row.borrow_mut();
                row.x = x;
                row.y = y;
                air.generate_subrow((x, y), (&mut row.inv, &mut row.out));
            });

        RowMajorMatrix::new(rows, width)
    }
}

#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_equal(x: u32, y: u32) {
    let x = PrimeCharacteristicRing::from_u32(x);
    let y = PrimeCharacteristicRing::from_u32(y);

    let chip = IsEqualChip {
        pairs: vec![(x, y)],
    };

    let trace = chip.generate_trace();

    let traces = vec![trace]
        .iter()
        .map(ColMajorMatrix::from_row_major)
        .map(AirProvingContext::simple_no_pis)
        .collect::<Vec<_>>();

    test_engine_small()
        .run_test(any_air_arc_vec![IsEqTestAir(IsEqSubAir)], traces)
        .expect("Verification failed");
}

#[test_matrix(
    [0,97,127],
    [0,23,97]
)]
fn test_single_is_zero_fail(x: u32, y: u32) {
    let x = PrimeCharacteristicRing::from_u32(x);
    let y = PrimeCharacteristicRing::from_u32(y);

    let chip = IsEqualChip {
        pairs: vec![(x, y)],
    };

    let mut trace = chip.generate_trace();
    trace.values[2] = if trace.values[2] == PrimeCharacteristicRing::ONE {
        PrimeCharacteristicRing::ZERO
    } else {
        PrimeCharacteristicRing::ONE
    };

    disable_debug_builder();
    let traces = vec![trace]
        .iter()
        .map(ColMajorMatrix::from_row_major)
        .map(AirProvingContext::simple_no_pis)
        .collect::<Vec<_>>();
    assert!(
        test_engine_small()
            .run_test(any_air_arc_vec![IsEqTestAir(IsEqSubAir)], traces)
            .is_err(),
        "Expected constraint to fail"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_is_equal_against_cpu_full() {
    let mut rng = create_seeded_rng();

    for log_height in 1..=16 {
        let n = 1 << log_height;

        let vec_x: Vec<F> = (0..n)
            .map(|_| F::from_u32(rng.random_range(0..F::ORDER_U32)))
            .collect();

        let vec_y: Vec<F> = (0..n)
            .map(|i| {
                if rng.random_bool(0.5) {
                    vec_x[i] // 50 % chance: equal to x
                } else {
                    F::from_u32(rng.random_range(0..F::ORDER_U32)) // 50% chance to be random
                }
            })
            .collect();

        let inputs_x = vec_x.as_slice().to_device().unwrap();
        let inputs_y = vec_y.as_slice().to_device().unwrap();

        let gpu_matrix = DeviceMatrix::<F>::with_capacity(n, 2);
        unsafe {
            is_equal::dummy_tracegen(gpu_matrix.buffer(), &inputs_x, &inputs_y).unwrap();
        }

        let cpu_matrix = Arc::new(RowMajorMatrix::<F>::new(
            (0..n)
                .flat_map(|i| {
                    let cur_x = vec_x[i];
                    let cur_y = vec_y[i];

                    let mut cur_inv = F::ONE;
                    let mut cur_out = F::ONE;
                    IsEqSubAir.generate_subrow((cur_x, cur_y), (&mut cur_inv, &mut cur_out));

                    [cur_inv, cur_out]
                })
                .collect::<Vec<_>>(),
            2,
        ));

        assert_eq_host_and_device_matrix(cpu_matrix, &gpu_matrix);
    }
}
