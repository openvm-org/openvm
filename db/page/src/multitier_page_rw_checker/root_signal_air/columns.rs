use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootSignalCols<T> {
    pub air_id: T,
    pub mult: T,
    pub range: Option<(Vec<T>, Vec<T>)>,
}

impl<T> RootSignalCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, is_init: bool) -> Self
    where
        T: Clone,
    {
        if is_init {
            RootSignalCols {
                air_id: cols[0].clone(),
                mult: cols[1].clone(),
                range: None,
            }
        } else {
            RootSignalCols {
                air_id: cols[0].clone(),
                mult: cols[1].clone(),
                range: Some((
                    cols[2..2 + idx_len].to_vec(),
                    cols[2 + idx_len..2 + 2 * idx_len].to_vec(),
                )),
            }
        }
    }
}
