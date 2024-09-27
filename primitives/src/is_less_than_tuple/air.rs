use std::borrow::Borrow;

use afs_stark_backend::{interaction::InteractionBuilder, rap::BaseAirWithPublicValues};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{IsLessThanTupleAuxCols, IsLessThanTupleCols, IsLessThanTupleIoCols};
use crate::{
    is_equal_vec::IsEqualVecAir,
    is_less_than::{columns::IsLessThanIoCols, IsLessThanAir},
    sub_chip::{AirConfig, SubAir},
    var_range::bus::VariableRangeCheckerBus,
};

#[derive(Clone, Debug)]
pub struct IsLessThanTupleAir {
    /// The number of bits to decompose each number into, for less than checking
    pub range_max_bits: usize,
    /// IsLessThanAirs for each tuple element
    pub is_less_than_airs: Vec<IsLessThanAir>,
    /// IsEqualVecAirs
    pub is_equal_vec_air: IsEqualVecAir,
    // Better to store this separately to avoid re-allocating vectors each time
    pub limb_bits: Vec<usize>,
}

impl IsLessThanTupleAir {
    pub fn new(bus: VariableRangeCheckerBus, limb_bits: Vec<usize>) -> Self {
        let range_max_bits = bus.range_max_bits;
        let is_less_than_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanAir::new(bus, limb_bit))
            .collect::<Vec<_>>();

        Self {
            range_max_bits,
            is_less_than_airs,
            is_equal_vec_air: IsEqualVecAir::new(limb_bits.len()),
            limb_bits,
        }
    }

    pub fn tuple_len(&self) -> usize {
        self.is_less_than_airs.len()
    }

    /// FOR INTERNAL USE ONLY when this AIR is used as a sub-AIR but the comparators `x, y` are on different rows. See [IsLessThanAir::eval_without_interactions].
    ///
    /// Constrains that `x < y` lexicographically.
    pub(crate) fn eval_without_interactions<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanTupleIoCols<AB::Expr>,
        aux: IsLessThanTupleAuxCols<AB::Var>,
    ) {
        let x = io.x;
        let y = io.y;

        // here we constrain that less_than[i] indicates whether x[i] < y[i] using the IsLessThan subchip for each i
        for (i, lt_aux_cols) in aux.less_than_aux.into_iter().enumerate() {
            let x_val = x[i].clone();
            let y_val = y[i].clone();

            let lt_io_cols = IsLessThanIoCols::new(x_val, y_val, aux.less_than[i]);

            self.is_less_than_airs[i].conditional_eval_without_interactions(
                builder,
                lt_io_cols,
                lt_aux_cols,
                AB::F::one(),
            );
        }

        let mut prods = aux.is_equal_vec_aux.prods.clone();
        prods.push(aux.is_equal_out);
        let invs = aux.is_equal_vec_aux.invs.clone();

        // initialize prods[0] = is_equal(x[0], y[0])
        builder.assert_eq(
            prods[0] + (x[0].clone() - y[0].clone()) * invs[0],
            AB::Expr::one(),
        );

        for i in 0..x.len() {
            // constrain prods[i] = 0 if x[i] != y[i]
            builder.assert_zero(prods[i] * (x[i].clone() - y[i].clone()));
        }

        for i in 0..x.len() - 1 {
            // if prod[i] == 0 all after are 0
            builder.assert_eq(prods[i] * prods[i + 1], prods[i + 1]);
            // prods[i] == 1 forces prods[i+1] == is_equal(x[i+1], y[i+1])
            builder.assert_eq(
                prods[i + 1] + (x[i + 1].clone() - y[i + 1].clone()) * invs[i + 1],
                prods[i],
            );
        }

        let less_than_cumulative = aux.less_than_cumulative.clone();

        builder.assert_eq(less_than_cumulative[0], aux.less_than[0]);

        for i in 1..x.len() {
            // this constrains that less_than_cumulative[i] indicates whether the first i elements of x are less than
            // the first i elements of y, lexicographically
            // note that less_than_cumulative[i - 1] and prods[i - 1] are never both 1
            builder.assert_eq(
                less_than_cumulative[i],
                less_than_cumulative[i - 1] + aux.less_than[i] * prods[i - 1],
            );
        }

        // constrain that the tuple_less_than does indicate whether x < y, lexicographically
        builder.assert_eq(io.tuple_less_than, less_than_cumulative[x.len() - 1]);
    }
}

impl AirConfig for IsLessThanTupleAir {
    type Cols<T> = IsLessThanTupleCols<T>;
}

impl<F: Field> BaseAirWithPublicValues<F> for IsLessThanTupleAir {}
impl<F: Field> BaseAir<F> for IsLessThanTupleAir {
    fn width(&self) -> usize {
        IsLessThanTupleCols::<F>::width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for IsLessThanTupleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanTupleCols::<AB::Var>::from_slice(local, self);

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-chip with constraints to check whether one tuple is less than the another
impl<AB: InteractionBuilder> SubAir<AB> for IsLessThanTupleAir {
    type IoView = IsLessThanTupleIoCols<AB::Var>;
    type AuxView = IsLessThanTupleAuxCols<AB::Var>;

    // constrain that x < y lexicographically
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let io_exprs = IsLessThanTupleIoCols::<AB::Expr>::from(io);

        self.eval_interactions(builder, &aux.less_than_aux);
        self.eval_without_interactions(builder, io_exprs, aux);

        // Note: if we had called the individual IsLessThanAir sub-airs with `eval`, they
        // would have added in the interactions automatically. We didn't do that here because
        // we need the `eval_without_interactions` version in AssertSortedAir where the comparators
        // `x, y` are on different rows. The rust trait bounds of AB: AirBuilder vs
        // AB: InteractionBuilder make this complicated to do otherwise.
    }
}

impl IsLessThanTupleAir {
    /// Imposes the non-interaction constraints on all except the last row. This is
    /// intended for use when the comparators `x, y` are on adjacent rows.
    ///
    /// This function does also enable the interaction constraints _on every row_.
    /// The `eval_interactions` performs range checks on `lower_decomp` on every row, even
    /// though in this AIR the lower_decomp is not used on the last row.
    /// This simply means the trace generation must fill in the last row with numbers in
    /// range (e.g., with zeros)
    pub fn eval_when_transition<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanTupleIoCols<AB::Var>,
        aux: IsLessThanTupleAuxCols<AB::Var>,
    ) {
        let io_exprs = IsLessThanTupleIoCols::<AB::Expr>::from(io);

        self.eval_interactions(builder, &aux.less_than_aux);
        self.eval_without_interactions(&mut builder.when_transition(), io_exprs, aux);
    }
}
