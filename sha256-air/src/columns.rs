use afs_chips::utils::Word32;
use afs_derive::AlignedBorrow;

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
pub struct Sha256Cols<T> {
    pub io: Sha256IoCols<T>,
    pub aux: Sha256AuxCols<T>,
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
pub struct Sha256IoCols<T> {
    // Two field elements to hold one word (32bits) of input
    pub input: Word32<T>,
    // 16 field elements to hold one 256bit output
    pub output: [T; 16],
}

#[derive(AlignedBorrow, Default, Debug, Clone, Copy)]
pub struct Sha256AuxCols<T> {
    // The message schedule at the current round.
    pub w: Word32<T>,
    // Sliding window of the previous 16 w.
    pub w_prev: [T; 32],
    // The round constant.
    pub k: Word32<T>,
    // The working variables a~h.
    pub a: Word32<T>,
    pub b: Word32<T>,
    pub c: Word32<T>,
    pub d: Word32<T>,
    pub e: Word32<T>,
    pub f: Word32<T>,
    pub g: Word32<T>,
    pub h: Word32<T>,
    // The bit operation variables to compute w.
    pub s0: Word32<T>,
    pub s1: Word32<T>,
    // The variables to update working variables each round.
    pub cs1: Word32<T>, // S1 in pseudocode
    pub ch: Word32<T>,
    pub temp1: Word32<T>,
    pub cs0: Word32<T>, // S0 in pseudocode
    pub maj: Word32<T>,
    pub temp2: Word32<T>,
    // TODO: use Vec? Fix for now so I can use aligned borrow and skip implementing from slice.
    pub e_bits: [T; 32],
    // Hash values.
    pub h0: Word32<T>,
    pub h1: Word32<T>,
    pub h2: Word32<T>,
    pub h3: Word32<T>,
    pub h4: Word32<T>,
    pub h5: Word32<T>,
    pub h6: Word32<T>,
    pub h7: Word32<T>,
    // Control flow columns.
    pub idx: T,
    pub row_idx: T,
    pub is_block_start: T, // Bool, basically whether row_idx == 0, but equality is hard to condition on.
}

impl<T> Sha256Cols<T> {
    pub fn flatten(&self) -> Vec<T> {
        todo!()
    }
}
