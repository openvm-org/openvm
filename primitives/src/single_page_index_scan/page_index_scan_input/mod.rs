use derive_more::Display;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::range_gate::RangeCheckerGateChip;

pub mod air;
pub mod bridge;
pub mod columns;
pub mod trace;

pub use air::PageIndexScanInputAir;

#[derive(Default, Debug, Display, Clone, Deserialize, Serialize, PartialEq)]
pub enum Comp {
    #[default]
    Lt,
    Lte,
    Eq,
    Gte,
    Gt,
}

/// Given a fixed predicate of the form index OP x, where OP is one of {<, <=, =, >=, >}
/// and x is a private input, the PageIndexScanInputChip implements a chip such that the chip:
///
/// 1. Has public value x and OP given by cmp (Lt, Lte, Eq, Gte, or Gt)
/// 2. Sends all rows of the page that match the predicate index OP x where x is the public value
pub struct PageIndexScanInputChip {
    pub air: PageIndexScanInputAir,
    pub range_checker: Arc<RangeCheckerGateChip>,
    pub cmp: Comp,
}

impl PageIndexScanInputChip {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        page_bus_index: usize,
        idx_len: usize,
        data_len: usize,
        idx_limb_bits: usize,
        decomp: usize,
        range_checker: Arc<RangeCheckerGateChip>,
        cmp: Comp,
    ) -> Self {
        let air = PageIndexScanInputAir::new(
            page_bus_index,
            range_checker.bus_index(),
            idx_len,
            data_len,
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
