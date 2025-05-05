use core::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

/// Columns for a Poseidon2 AIR which computes one permutation per row.
///
/// The columns of the STARK are divided into the three different round sections of the Poseidon2
/// Permutation: beginning full rounds, partial rounds, and ending full rounds. For the full
/// rounds we store an [`SBox`] columnset for each state variable, and for the partial rounds we
/// store only for the first state variable. Because the matrix multiplications are linear
/// functions, we need only keep auxiliary columns for the S-box computations.
#[repr(C)]
pub struct MultiRowPoseidon2Cols<
    T,
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
> {
    // Only true if idx == 0
    // Also preprocessed
    pub export: T,
    pub is_beginning: T,
    pub is_partial: T,
    pub is_full: T,

    pub constants: [T; WIDTH],

    pub inputs: [T; WIDTH],

    pub cur_state: [T; WIDTH],

    pub end_state: [T; WIDTH],

    // The input is now degree 2 unfortunately.
    pub sbox: [SBoxPost<T, SBOX_DEGREE, SBOX_REGISTERS>; WIDTH],
}

/// Partial round columns.
#[repr(C)]
pub struct SBoxPost<T, const SBOX_DEGREE: u64, const SBOX_REGISTERS: usize> {
    /// Possible intermediate results within the S-box.
    pub sbox: SBox<T, SBOX_DEGREE, SBOX_REGISTERS>,
    /// The output of the S-box.
    pub post_sbox: T,
}

/// Possible intermediate results within an S-box.
///
/// Use this column-set for an S-box that can be computed with `REGISTERS`-many intermediate results
/// (not counting the final output). The S-box is checked to ensure that `REGISTERS` is the optimal
/// number of registers for the given `DEGREE` for the degrees given in the Poseidon2 paper:
/// `3`, `5`, `7`, and `11`. See `eval_sbox` for more information.
#[repr(C)]
pub struct SBox<T, const DEGREE: u64, const REGISTERS: usize>(pub [T; REGISTERS]);

pub const fn num_cols<
    const WIDTH: usize,
    const SBOX_DEGREE: u64,
    const SBOX_REGISTERS: usize,
    const HALF_FULL_ROUNDS: usize,
    const PARTIAL_ROUNDS: usize,
>() -> usize {
    size_of::<
        MultiRowPoseidon2Cols<
            u8,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    >()
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    Borrow<
        MultiRowPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    > for [T]
{
    fn borrow(
        &self,
    ) -> &MultiRowPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to::<MultiRowPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

impl<
        T,
        const WIDTH: usize,
        const SBOX_DEGREE: u64,
        const SBOX_REGISTERS: usize,
        const HALF_FULL_ROUNDS: usize,
        const PARTIAL_ROUNDS: usize,
    >
    BorrowMut<
        MultiRowPoseidon2Cols<
            T,
            WIDTH,
            SBOX_DEGREE,
            SBOX_REGISTERS,
            HALF_FULL_ROUNDS,
            PARTIAL_ROUNDS,
        >,
    > for [T]
{
    fn borrow_mut(
        &mut self,
    ) -> &mut MultiRowPoseidon2Cols<
        T,
        WIDTH,
        SBOX_DEGREE,
        SBOX_REGISTERS,
        HALF_FULL_ROUNDS,
        PARTIAL_ROUNDS,
    > {
        // debug_assert_eq!(self.len(), NUM_COLS);
        let (prefix, shorts, suffix) = unsafe {
            self.align_to_mut::<MultiRowPoseidon2Cols<
                T,
                WIDTH,
                SBOX_DEGREE,
                SBOX_REGISTERS,
                HALF_FULL_ROUNDS,
                PARTIAL_ROUNDS,
            >>()
        };
        debug_assert!(prefix.is_empty(), "Alignment should match");
        debug_assert!(suffix.is_empty(), "Alignment should match");
        debug_assert_eq!(shorts.len(), 1);
        &mut shorts[0]
    }
}
