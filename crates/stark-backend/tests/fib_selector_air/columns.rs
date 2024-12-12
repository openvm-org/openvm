use openvm_circuit_primitives_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FibonacciSelectorCols<F> {
    pub sel: F,
}
