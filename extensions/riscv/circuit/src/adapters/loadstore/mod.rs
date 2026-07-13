use openvm_circuit::arch::BLOCK_FE_WIDTH;
use openvm_circuit_primitives::encoder::Encoder;

mod load_aligned;
mod load_misaligned;
mod store_aligned;
mod store_misaligned;

pub use load_aligned::*;
pub use load_misaligned::*;
pub use store_aligned::*;
pub use store_misaligned::*;

/// Load/store memory access widths in bytes.
pub(crate) const LOAD_WIDTH_BYTE: usize = 1;
pub(crate) const LOAD_WIDTH_HALFWORD: usize = 2;
pub(crate) const LOAD_WIDTH_WORD: usize = 4;
pub(crate) const LOAD_WIDTH_DOUBLEWORD: usize = 8;
pub(crate) const STORE_WIDTH_BYTE: usize = 1;
pub(crate) const STORE_WIDTH_HALFWORD: usize = 2;
pub(crate) const STORE_WIDTH_WORD: usize = 4;
pub(crate) const STORE_WIDTH_DOUBLEWORD: usize = 8;

/// Byte shifts of an effective pointer inside an 8-byte memory block. Every load/store core
/// encodes shift `i` as selector case `i`.
pub(crate) const NUM_BYTE_SHIFTS: usize = 2 * BLOCK_FE_WIDTH;
/// Maximal degree of the load/store shift-selector flag expressions.
const SHIFT_SELECTOR_MAX_DEGREE: u32 = 2;

/// Selector encoder shared by all load/store cores: one case per byte shift, with the zero
/// point reserved for invalid rows.
pub(crate) fn shift_encoder<const SELECTOR_WIDTH: usize>() -> Encoder {
    let encoder = Encoder::new(NUM_BYTE_SHIFTS, SHIFT_SELECTOR_MAX_DEGREE, true);
    debug_assert_eq!(encoder.width(), SELECTOR_WIDTH);
    encoder
}
