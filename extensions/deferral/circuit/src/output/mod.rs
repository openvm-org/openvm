use openvm_circuit::arch::{ExecutionError, VmChipWrapper};

mod air;
#[cfg(feature = "cuda")]
mod cuda;
mod execution;
mod trace;

pub use air::*;
#[cfg(feature = "cuda")]
pub use cuda::*;
pub use trace::*;

#[cfg(test)]
mod tests;

pub type DeferralOutputChip<F> = VmChipWrapper<F, DeferralOutputFiller<F>>;

#[inline(always)]
fn checked_deferral_index(
    pc: u32,
    num_deferrals: usize,
    deferral_idx: u32,
) -> Result<usize, ExecutionError> {
    let deferral_idx = deferral_idx as usize;
    if deferral_idx < num_deferrals {
        Ok(deferral_idx)
    } else {
        Err(ExecutionError::Fail {
            pc,
            msg: "deferral index is out of bounds",
        })
    }
}
