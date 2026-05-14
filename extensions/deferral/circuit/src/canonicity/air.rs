use itertools::{izip, Itertools};
use openvm_circuit_primitives::{utils::not, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};

use super::{CanonicityAuxCols, CanonicityIo, CANONICITY_LIMB_BITS, CANONICITY_NUM_LIMBS};

/// Sub-AIR to constrain that a field limb decomposition is canonical. Note:
/// - It is assumed that each limb has been range checked to `CANONICITY_LIMB_BITS`.
/// - `eval` returns a value to be range checked (in `[0, 2^CANONICITY_LIMB_BITS)`).
///
/// The lexicographic comparison is granularity-agnostic — it works against
/// any limb decomposition of `F::ORDER_U32`. For the u16-cell migration the
/// AIR walks 2 u16 limbs MSL-first; before the migration it walked 4 bytes.
pub struct CanonicitySubAir;

impl<AB: InteractionBuilder> SubAir<AB> for CanonicitySubAir
where
    AB::F: PrimeField32,
{
    type AirContext<'a>
        = (
        CanonicityIo<AB::Expr>,
        &'a CanonicityAuxCols<AB::Var>,
        &'a mut AB::Expr,
    )
    where
        AB::Expr: 'a,
        AB::Var: 'a,
        AB: 'a;

    fn eval<'a>(
        &'a self,
        builder: &'a mut AB,
        (io, aux, to_range_check): (
            CanonicityIo<AB::Expr>,
            &'a CanonicityAuxCols<AB::Var>,
            &'a mut AB::Expr,
        ),
    ) where
        AB::Var: 'a,
        AB::Expr: 'a,
    {
        // Most-significant-limb-first decomposition of `F::ORDER_U32` into
        // `CANONICITY_NUM_LIMBS` chunks of `CANONICITY_LIMB_BITS` bits. We rely
        // on the AB::F prime fitting in `CANONICITY_NUM_LIMBS * CANONICITY_LIMB_BITS`
        // bits; for BabyBear (31 bits) and `2 * 16 = 32 >= 31` this is fine.
        debug_assert!(
            (CANONICITY_NUM_LIMBS as u32) * (CANONICITY_LIMB_BITS as u32)
                >= 32 - (AB::F::ORDER_U32.leading_zeros()),
            "CANONICITY_NUM_LIMBS * CANONICITY_LIMB_BITS must cover F::ORDER_U32"
        );
        let limb_mask: u32 = if CANONICITY_LIMB_BITS == 32 {
            u32::MAX
        } else {
            (1u32 << CANONICITY_LIMB_BITS) - 1
        };
        let order_be = (0..CANONICITY_NUM_LIMBS).rev().map(|i| {
            AB::Expr::from_u32((AB::F::ORDER_U32 >> (i * CANONICITY_LIMB_BITS)) & limb_mask)
        });

        let mut prefix_sum = AB::Expr::ZERO;

        for (x, y, &marker) in izip!(io.x, order_be, aux.diff_marker.iter()) {
            let diff = y - x;
            prefix_sum += marker.into();
            builder.assert_bool(marker);
            builder.when(marker).assert_one(io.count.clone());
            builder
                .when(io.count.clone())
                .assert_zero(not::<AB::Expr>(prefix_sum.clone()) * diff.clone());
            builder.when(marker).assert_eq(aux.diff_val, diff);
        }

        builder.assert_bool(prefix_sum.clone());
        builder.when(io.count.clone()).assert_one(prefix_sum);

        *to_range_check = AB::Expr::from(aux.diff_val) - AB::Expr::ONE;
    }
}

impl CanonicitySubAir {
    pub fn assert_canonicity<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        x: &[AB::Var],
        aux: &CanonicityAuxCols<AB::Var>,
        enabled: AB::Expr,
    ) -> AB::Expr
    where
        AB::F: PrimeField32,
    {
        let io = CanonicityIo {
            x: x.iter().rev().map(|b| (*b).into()).collect_array().unwrap(),
            count: enabled,
        };
        let mut ret = AB::Expr::ZERO;
        self.eval(builder, (io, aux, &mut ret));
        ret
    }
}
