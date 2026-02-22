use std::{
    borrow::{Borrow, BorrowMut},
    sync::atomic::{AtomicU32, Ordering},
};

use openvm_circuit::utils::next_power_of_two_or_zero;
use openvm_circuit_primitives::{utils::not, Chip};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::{AirProvingContext, ColMajorMatrix, CpuBackend},
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, Val,
};

pub mod bus;
use bus::*;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralCircuitCountCols<T> {
    pub is_valid: T,
    pub row_idx: T,
    pub mult: T,
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralCircuitCountAir {
    pub lookup_bus: DeferralCircuitCountBus,
    pub num_deferral_circuits: usize,
}

impl<F> BaseAir<F> for DeferralCircuitCountAir {
    fn width(&self) -> usize {
        DeferralCircuitCountCols::<F>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralCircuitCountAir {}
impl<F> PartitionedBaseAir<F> for DeferralCircuitCountAir {}

impl<AB> Air<AB> for DeferralCircuitCountAir
where
    AB: InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("row 0 present");
        let next = main.row_slice(1).expect("row 1 present");

        let local: &DeferralCircuitCountCols<AB::Var> = (*local).borrow();
        let next: &DeferralCircuitCountCols<AB::Var> = (*next).borrow();

        // Base constraints to ensure all valid rows are at the beginning of the trace,
        // that row_idx increments properly, and that mult is 0 on invalid rows.
        builder.assert_bool(local.is_valid);
        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);

        builder.when_first_row().assert_zero(local.row_idx);
        builder
            .when_transition()
            .assert_one(next.row_idx - local.row_idx);

        builder.when(not(local.is_valid)).assert_zero(local.mult);

        // Constrain that there are exactly n = num_deferral_circuits valid rows. We do
        // this by constraining that local.is_valid equals next.is_valid on rows such
        // that row_idx + 1 != n, and then either that the last row is invalid (meaning
        // the last valid row was n - 1) or that n is the trace height.
        let num_valid = AB::F::from_usize(self.num_deferral_circuits);

        builder
            .when_transition()
            .when_ne(next.row_idx, num_valid)
            .assert_eq(local.is_valid, next.is_valid);
        builder
            .when_last_row()
            .when(local.is_valid)
            .assert_eq(local.row_idx + AB::Expr::ONE, num_valid);

        // Provide the lookup for valid deferral indices.
        self.lookup_bus
            .receive(local.row_idx)
            .eval(builder, local.mult);
    }
}

#[derive(Debug)]
pub struct DeferralCircuitCountChip {
    pub count: Vec<AtomicU32>,
}

impl DeferralCircuitCountChip {
    pub fn new(num_deferral_circuit: usize) -> Self {
        let count = (0..num_deferral_circuit)
            .map(|_| AtomicU32::new(0))
            .collect();
        Self { count }
    }

    pub fn add_count(&self, idx: u32) {
        let idx = idx as usize;
        assert!(idx < self.count.len());
        let val_atomic = &self.count[idx];
        val_atomic.fetch_add(1, Ordering::Relaxed);
    }
}

impl<SC: StarkProtocolConfig, RA> Chip<RA, CpuBackend<SC>> for DeferralCircuitCountChip
where
    Val<SC>: PrimeCharacteristicRing,
{
    fn generate_proving_ctx(&self, _: RA) -> AirProvingContext<CpuBackend<SC>> {
        let width = DeferralCircuitCountCols::<u8>::width();
        let height = next_power_of_two_or_zero(self.count.len());
        let mut trace = vec![Val::<SC>::ZERO; width * height];

        let mut rows = trace.chunks_exact_mut(width);
        let mut row_idx = 0u32;

        for mult in &self.count {
            let row = rows.next().unwrap();
            let cols: &mut DeferralCircuitCountCols<Val<SC>> = (*row).borrow_mut();
            cols.is_valid = Val::<SC>::ONE;
            cols.row_idx = Val::<SC>::from_u32(row_idx);
            cols.mult = Val::<SC>::from_u32(mult.swap(0, Ordering::Relaxed));
            row_idx += 1;
        }

        for row in rows {
            let cols: &mut DeferralCircuitCountCols<Val<SC>> = (*row).borrow_mut();
            cols.row_idx = Val::<SC>::from_u32(row_idx);
            row_idx += 1;
        }

        let trace = RowMajorMatrix::new(trace, width);
        AirProvingContext::simple_no_pis(ColMajorMatrix::from_row_major(&trace))
    }
}
