//! Implementation of the SHA256/SHA512 compression function without padding
//! This this AIR doesn't constrain any of the message padding

mod air;
mod columns;
mod config;
mod trace;
mod utils;

pub use air::*;
pub use columns::*;
pub use config::*;
pub use trace::*;
pub use utils::*;

#[cfg(test)]
mod tests;
