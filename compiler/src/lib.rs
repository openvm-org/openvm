#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;

pub mod asm;
pub mod constraints;
pub mod ir;
pub mod util;
pub mod conversion;

pub mod prelude {
    pub use crate::asm::AsmCompiler;
    pub use crate::ir::*;
}
