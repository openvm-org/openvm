mod byte;
mod multi_byte;

pub use byte::*;
pub use multi_byte::*;

pub(crate) const LOAD_WIDTH_BYTE: usize = 1;
pub(crate) const LOAD_WIDTH_HALFWORD: usize = 2;
pub(crate) const LOAD_WIDTH_WORD: usize = 4;
pub(crate) const LOAD_WIDTH_DOUBLEWORD: usize = 8;
