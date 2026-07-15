//! F-typed builder for the rvr deferral output hasher.

use openvm_circuit::arch::VmField;
use rvr_openvm_ext_deferral::{DeferralCompressFn, DeferralHashFn, DEFERRAL_COMMIT_NUM_BYTES};

use crate::{def_fn::hash_output_raw, poseidon2::deferral_poseidon2_chip};

pub fn make_deferral_hash<F: VmField>() -> DeferralHashFn {
    let hasher = deferral_poseidon2_chip::<F>();
    Box::new(
        move |def_idx: u32, output_raw: &[u8]| -> [u8; DEFERRAL_COMMIT_NUM_BYTES] {
            hash_output_raw(&hasher, def_idx, output_raw)
                .as_slice()
                .try_into()
                .unwrap()
        },
    )
}

/// Builds the accumulator compression used by RVR deferral CALL.
/// Inputs and outputs are canonical u32 field values.
pub fn make_deferral_compress<F: VmField>() -> DeferralCompressFn {
    let hasher = deferral_poseidon2_chip::<F>();
    Box::new(move |lhs, rhs| {
        let lhs_f = lhs.map(F::from_u32);
        let rhs_f = rhs.map(F::from_u32);
        hasher
            .perm(&lhs_f, &rhs_f, true)
            .map(|v| v.as_canonical_u32())
    })
}
