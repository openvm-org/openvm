use afs_middleware_derive::AlignedBorrow;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FibonacciSelectorCols<F> {
    pub sel: F,
}
