/// This chip is mostly just FinalPageAir, but with different interactions
/// Most of the code for it is here (just calls the correspondsing functions
/// from FinalPageAir), but the new interactions are in bridge.rs
use std::sync::Arc;

use afs_stark_backend::air_builders::PartitionedAirBuilder;
use p3_air::{Air, BaseAir};
use p3_field::{Field, PrimeField};
use p3_matrix::dense::RowMajorMatrix;
use p3_uni_stark::{StarkGenericConfig, Val};

use crate::{
    final_page::{columns::FinalPageCols, FinalPageAir},
    range_gate::RangeCheckerGateChip,
    sub_chip::AirConfig,
};

pub mod bridge;

#[derive(Clone)]
pub struct MyFinalTableAir {
    t1_output_bus_index: usize,
    t2_output_bus_index: usize,

    /// Foreign key indices within the data
    fkey_start: usize,
    fkey_end: usize,

    t2_data_len: usize,

    final_air: FinalPageAir,
}

impl MyFinalTableAir {
    pub fn new(
        t1_output_bus_index: usize,
        t2_output_bus_index: usize,
        range_bus_index: usize,
        idx_len: usize,
        t1_data_len: usize,
        t2_data_len: usize,
        fkey_start: usize,
        fkey_end: usize,
        idx_limb_bits: usize,
        idx_decomp: usize,
    ) -> Self {
        Self {
            t1_output_bus_index,
            t2_output_bus_index,
            fkey_start,
            fkey_end,
            t2_data_len,
            final_air: FinalPageAir::new(
                range_bus_index,
                idx_len,
                t1_data_len + t2_data_len,
                idx_limb_bits,
                idx_decomp,
            ),
        }
    }

    pub fn air_width(&self) -> usize {
        self.final_air.air_width()
    }

    // Trace generation
    pub fn gen_page_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
    ) -> RowMajorMatrix<Val<SC>> {
        self.final_air.gen_page_trace::<SC>(page)
    }

    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: Vec<Vec<u32>>,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air
            .gen_aux_trace::<SC>(page.clone(), range_checker)
    }
}

impl<F: Field> BaseAir<F> for MyFinalTableAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl AirConfig for MyFinalTableAir {
    type Cols<T> = FinalPageCols<T>;
}

impl<AB: PartitionedAirBuilder> Air<AB> for MyFinalTableAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // Making sure the page is in the proper format
        Air::eval(&self.final_air, builder);
    }
}
