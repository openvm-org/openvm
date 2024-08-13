pub mod afs_exec;
pub mod afs_expr;
pub mod afs_node;
pub mod committed_page;

pub static BITS_PER_FE: usize = 16;
pub static MAX_ROWS: usize = 16;
pub static PCS_LOG_DEGREE: usize = 4;
pub static RANGE_CHECK_BITS: usize = 16;

pub static PAGE_BUS_IDX: usize = 0;
pub static RANGE_BUS_IDX: usize = 1;
pub static OPS_BUS_IDX: usize = 2;
