use itertools::izip;
use openvm_circuit_primitives::{utils::not, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};

use super::{CanonicityAuxCols, CanonicityIo};
use crate::utils::{F_NUM_U16S, U16_BITS, U16_MASK};

/// Sub-AIR to constrain that a field limb decomposition is canonical. Note:
/// - It is assumed that each limb has been range checked to `U16_BITS`.
/// - `eval` returns a value to be range checked (in `[0, 2^U16_BITS)`).
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
        // Decompose `F::ORDER_U32` into the configured MSL-first limb shape.
        debug_assert!(
            (F_NUM_U16S as u32) * (U16_BITS as u32) >= 32 - (AB::F::ORDER_U32.leading_zeros()),
            "F_NUM_U16S * U16_BITS must cover F::ORDER_U32"
        );
        let order_be = (0..F_NUM_U16S)
            .rev()
            .map(|i| AB::Expr::from_u32((AB::F::ORDER_U32 >> (i * U16_BITS)) & U16_MASK));

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
    /// Constrain canonicity of the `F_NUM_U16S` LE u16-cell limbs in `x`.
    /// `x` is given LE; the sub-AIR consumes MSL-first, so we reverse here.
    pub fn assert_canonicity<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        x: [AB::Expr; F_NUM_U16S],
        aux: &CanonicityAuxCols<AB::Var>,
        enabled: AB::Expr,
    ) -> AB::Expr
    where
        AB::F: PrimeField32,
    {
        let mut x_be = x;
        x_be.reverse();
        let io = CanonicityIo {
            x: x_be,
            count: enabled,
        };
        let mut ret = AB::Expr::ZERO;
        self.eval(builder, (io, aux, &mut ret));
        ret
    }
}
