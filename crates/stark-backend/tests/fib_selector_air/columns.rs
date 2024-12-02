use ax_circuit_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FibonacciSelectorCols<F> {
    pub sel: F,
}
