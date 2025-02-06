use openvm_sha256_air::{Sha256Config, ShaConfig};
use openvm_sha_macros::ColsRef;

#[repr(C)]
#[derive(Clone, Copy, Debug, ColsRef)]
pub struct ShaFlagsCols<T, const ROW_VAR_CNT: usize> {
    pub is_round_row: T,
    /// A flag that indicates if the current row is among the first 4 rows of a block
    pub is_first_4_rows: T,
    pub is_digest_row: T,
    pub is_last_block: T,
    /// We will encode the row index [0..17) using 5 cells
    //#[length(ROW_VAR_CNT)]
    pub row_idx: [T; ROW_VAR_CNT],
    /// The global index of the current block
    pub global_block_idx: T,
    /// Will store the index of the current block in the current message starting from 0
    pub local_block_idx: T,
}

#[test]
fn flags() {
    let input = [1; 4 + 5 + 2];
    let test: ShaFlagsColsRef<u32> = ShaFlagsColsRef::from::<Sha256Config>(&input);
    println!(
        "{}\n{}\n{}\n{}\n{:?}\n{}\n{}",
        test.is_round_row,
        test.is_first_4_rows,
        test.is_digest_row,
        test.is_last_block,
        test.row_idx,
        test.global_block_idx,
        test.local_block_idx
    );
}
