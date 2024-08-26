use std::sync::Arc;

use afs_primitives::range_gate::RangeCheckerGateChip;

use self::air::FilterInputTableAir;
use crate::common::comp::Comp;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub struct FilterInputTableChip {
    pub air: FilterInputTableAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub cmp: Comp,
}

impl FilterInputTableChip {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        start_col: usize,
        end_col: usize,
        idx_limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
        cmp: Comp,
    ) -> Self {
        let air = FilterInputTableAir::new(
            page_bus_index,
            range_checker.bus_index(),
            idx_len,
            data_len,
            start_col,
            end_col,
            idx_limb_bits,
            decomp,
            cmp.clone(),
        );

        Self {
            air,
            range_checker,
            cmp,
        }
    }
}
