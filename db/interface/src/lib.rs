pub mod common;
pub mod controller;
pub mod node;
pub mod utils;

/// Number of columns, from the left side of the Schema, that are index columns. Keep in mind that you
/// will also need to change the underlying `Page` data's idx cols to match this.
pub static NUM_IDX_COLS: usize = 1;

pub static BITS_PER_FE: usize = 16;
pub static MAX_ROWS: usize = 64;
pub static PCS_LOG_DEGREE: usize = 16;
pub static RANGE_CHECK_BITS: usize = 8;

pub static PAGE_BUS_IDX: usize = 0;
pub static RANGE_BUS_IDX: usize = 1;
pub static OPS_BUS_IDX: usize = 2;
