#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;
extern crate core;

pub mod asm;
pub mod constraints;
pub mod conversion;
pub mod ir;

pub mod prelude {
    pub use axvm_native_compiler_derive::{DslVariable, Hintable};

    pub use crate::{asm::AsmCompiler, ir::*};
}
