use itertools::{izip, Itertools};
use openvm_circuit_primitives::{utils::not, SubAir};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{PrimeCharacteristicRing, PrimeField32},
};

use super::{CanonicityAuxCols, CanonicityIo};

/// Sub-AIR to constrain that a field byte decomposition is canonical. Note:
/// - It is assumed that each value has been range checked
/// - eval returns a value to be range check
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
        let order_be = AB::F::ORDER_U32
            .to_le_bytes()
            .into_iter()
            .rev()
            .map(AB::Expr::from_u8);

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
