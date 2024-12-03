#[cfg(target_os = "zkvm")]
compile_error!("This crate should not be used with axvm target");

pub use axvm_ecc_guest;

pub mod curves;

#[cfg(test)]
mod tests;
