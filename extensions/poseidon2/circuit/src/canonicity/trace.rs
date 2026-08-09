use std::array::from_fn;

use openvm_stark_backend::p3_field::PrimeField32;

use super::{CanonicityAuxCols, F_NUM_BYTES};

/// Tracegen helper for canonicity sub-AIR auxiliary columns
pub struct CanonicityTraceGen;

impl CanonicityTraceGen {
    /// Fills `aux` for the little-endian byte decomposition `x_le` and returns the value that the
    /// caller must send to an 8-bit range checker.
    pub fn generate_subrow<F: PrimeField32>(
        x_le: &[F; F_NUM_BYTES],
        aux: &mut CanonicityAuxCols<F>,
    ) -> u32 {
        aux.diff_marker.fill(F::ZERO);
        aux.diff_val = F::ZERO;
        let x_be: [F; F_NUM_BYTES] = from_fn(|i| x_le[F_NUM_BYTES - 1 - i]);
        let order_be = F::ORDER_U32.to_le_bytes().into_iter().rev();

        let mut found = false;
        let mut to_range_check = 0u32;

        for (i, (&x, y)) in x_be.iter().zip(order_be).enumerate() {
            let x_u32 = x.as_canonical_u32();
            if !found && x_u32 != y as u32 {
                debug_assert!(x_u32 < 256);
                debug_assert!(y as u32 > x_u32);
                let diff = y as u32 - x_u32;
                aux.diff_marker[i] = F::ONE;
                aux.diff_val = F::from_u32(diff);
                to_range_check = diff - 1;
                found = true;
            }
        }
        debug_assert!(found);
        to_range_check
    }
}
