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
    common::page::Page, indexed_output_page_air::IndexedOutputPageAir,
    range_gate::RangeCheckerGateChip,
};

use super::controller::{T2Format, TableFormat};

pub mod bridge;

#[derive(Clone, derive_new::new)]
pub(super) struct FinalTableBuses {
    t1_output_bus_index: usize,
    t2_output_bus_index: usize,
}

#[derive(Clone)]
pub(super) struct FinalTableAir {
    buses: FinalTableBuses,
    /// Foreign key start index within the data
    fkey_start: usize,
    /// Foreign key end index within the data
    fkey_end: usize,
    t2_data_len: usize,

    final_air: IndexedOutputPageAir,
}

impl FinalTableAir {
    pub fn new(
        buses: FinalTableBuses,
        range_bus_index: usize,
        t1_format: TableFormat,
        t2_format: T2Format,
        decomp: usize,
    ) -> Self {
        Self {
            buses,
            fkey_start: t2_format.fkey_start,
            fkey_end: t2_format.fkey_end,
            t2_data_len: t2_format.table_format.data_len,
            final_air: IndexedOutputPageAir::new(
                range_bus_index,
                t2_format.table_format.idx_len,
                t1_format.data_len + t2_format.table_format.data_len,
                t2_format.table_format.idx_limb_bits,
                decomp,
            ),
        }
    }

    pub fn table_width(&self) -> usize {
        self.final_air.page_width()
    }

    pub fn aux_width(&self) -> usize {
        self.final_air.aux_width()
    }

    pub fn air_width(&self) -> usize {
        self.final_air.air_width()
    }

    pub fn gen_aux_trace<SC: StarkGenericConfig>(
        &self,
        page: &Page,
        range_checker: Arc<RangeCheckerGateChip>,
    ) -> RowMajorMatrix<Val<SC>>
    where
        Val<SC>: PrimeField,
    {
        self.final_air.gen_aux_trace::<SC>(page, range_checker)
    }
}

impl<F: Field> BaseAir<F> for FinalTableAir {
    fn width(&self) -> usize {
        self.air_width()
    }
}

impl<AB: PartitionedAirBuilder> Air<AB> for FinalTableAir
where
    AB::M: Clone,
{
    fn eval(&self, builder: &mut AB) {
        // Making sure the page is in the proper format
        Air::eval(&self.final_air, builder);
    }
}
