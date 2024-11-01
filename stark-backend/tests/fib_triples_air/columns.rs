use ax_circuit_derive::AlignedBorrow;

pub const NUM_FIBONACCI_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FibonacciCols<F> {
    pub left: F,
    pub middle: F,
    pub right: F,
}

impl<F> FibonacciCols<F> {
    pub const fn new(left: F, middle: F, right: F) -> FibonacciCols<F> {
        FibonacciCols {
            left,
            middle,
            right,
        }
    }
}
