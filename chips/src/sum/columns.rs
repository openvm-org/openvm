use afs_derive::AlignedBorrow;

pub const NUM_SUM_GATE_COLS: usize = 2;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SumGateCols<F> {
    pub input: F,
    pub partial_sum: F,
}

impl<F> SumGateCols<F> {
    pub const fn new(input: F, partial_sum: F) -> SumGateCols<F> {
        SumGateCols { input, partial_sum }
    }
}
