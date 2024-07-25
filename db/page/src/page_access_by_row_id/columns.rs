use afs_derive::AlignedBorrow;

#[derive(Clone, Debug)]
pub struct PageAccessByRowIdCols<T> {
    /// In trace partition 0, the page blob itself.
    pub page: Vec<T>,

    /// In trace partition 1
    pub aux: PageAccessByRowIdAuxCols<T>,
}

#[derive(Clone, Copy, Debug, AlignedBorrow)]
#[repr(C)]
pub struct PageAccessByRowIdAuxCols<T> {
    /// In trace partition 1
    pub row_id: T,
    /// In trace partition 1
    pub mult: T,
}
