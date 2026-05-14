use std::array::from_fn;

use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};

use super::{CanonicityAuxCols, CANONICITY_LIMB_BITS, CANONICITY_NUM_LIMBS};

/// Tracegen helper for canonicity sub-AIR auxiliary columns.
pub struct CanonicityTraceGen;

impl CanonicityTraceGen {
    /// Walk `x_le` MSL-first against the limb decomposition of `F::ORDER_U32`
    /// and populate `aux`. Returns the value `diff - 1` that the caller must
    /// range-check to `CANONICITY_LIMB_BITS`.
    pub fn generate_subrow<F: PrimeField32>(
        x_le: &[F; CANONICITY_NUM_LIMBS],
        aux: &mut CanonicityAuxCols<F>,
    ) -> u32 {
        aux.diff_marker.fill(F::ZERO);
        aux.diff_val = F::ZERO;
        let x_be: [F; CANONICITY_NUM_LIMBS] = from_fn(|i| x_le[CANONICITY_NUM_LIMBS - 1 - i]);

        let limb_mask: u32 = if CANONICITY_LIMB_BITS == 32 {
            u32::MAX
        } else {
            (1u32 << CANONICITY_LIMB_BITS) - 1
        };

        let mut found = false;
        let mut to_range_check = 0u32;

        for (i, &x) in x_be.iter().enumerate() {
            let limb_idx = CANONICITY_NUM_LIMBS - 1 - i;
            let y = (F::ORDER_U32 >> (limb_idx * CANONICITY_LIMB_BITS)) & limb_mask;
            let x_u32 = x.as_canonical_u32();
            if !found && x_u32 != y {
                debug_assert!(x_u32 < (1u64 << CANONICITY_LIMB_BITS) as u32);
                debug_assert!(y > x_u32);
                let diff = y - x_u32;
                aux.diff_marker[i] = F::ONE;
                aux.diff_val = F::from_u32(diff);
                to_range_check = diff - 1;
                found = true;
            }
        }
        debug_assert!(found);
        to_range_check
    }

    pub fn clear_aux<F: PrimeCharacteristicRing>(aux: &mut CanonicityAuxCols<F>) {
        aux.diff_marker.fill(F::ZERO);
        aux.diff_val = F::ZERO;
    }
}
