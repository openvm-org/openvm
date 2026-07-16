mod aligned;
mod misaligned;

pub use aligned::*;
pub use misaligned::*;

pub(crate) const LOAD_WIDTH_BYTE: usize = 1;
pub(crate) const LOAD_WIDTH_HALFWORD: usize = 2;
pub(crate) const LOAD_WIDTH_WORD: usize = 4;
pub(crate) const LOAD_WIDTH_DOUBLEWORD: usize = 8;
