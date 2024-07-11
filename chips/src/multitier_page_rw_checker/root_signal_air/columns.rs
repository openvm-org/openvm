use afs_derive::AlignedBorrow;

pub const NUM_COLS: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct RootSignalCols<T> {
    pub root_commitment: Vec<T>,
    pub air_id: T,
    pub mult: T,
    pub range: Option<(Vec<T>, Vec<T>)>,
}

impl<T> RootSignalCols<T> {
    pub fn from_slice(cols: &[T], idx_len: usize, commitment_len: usize, is_init: bool) -> Self
    where
        T: Clone,
    {
        if is_init {
            RootSignalCols {
                root_commitment: cols[0..commitment_len].to_vec(),
                air_id: cols[commitment_len].clone(),
                mult: cols[commitment_len + 1].clone(),
                range: None,
            }
        } else {
            RootSignalCols {
                root_commitment: cols[0..commitment_len].to_vec(),
                air_id: cols[commitment_len].clone(),
                mult: cols[commitment_len + 1].clone(),
                range: Some((
                    cols[commitment_len + 2..commitment_len + 2 + idx_len].to_vec(),
                    cols[commitment_len + 2 + idx_len..commitment_len + 2 + 2 * idx_len].to_vec(),
                )),
            }
        }
    }
}
