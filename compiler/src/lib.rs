#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;

pub mod asm;
pub mod constraints;
pub mod conversion;
pub mod ir;
pub mod util;

pub mod prelude {
    pub use afs_derive::DslVariable;

    pub use crate::{asm::AsmCompiler, ir::*};
}
