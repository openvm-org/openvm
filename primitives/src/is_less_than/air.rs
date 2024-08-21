use std::borrow::Borrow;

use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{AbstractField, Field};
use p3_matrix::Matrix;

use super::columns::{IsLessThanAuxCols, IsLessThanCols, IsLessThanIoCols};
use crate::sub_chip::{AirConfig, SubAir};

#[derive(Copy, Clone, Debug)]
pub struct IsLessThanAir {
    /// The bus index for sends to range chip
    pub bus_index: usize,
    /// The maximum number of bits for the numbers to compare
    pub max_bits: usize,
    /// The number of bits to decompose each number into, for less than checking
    pub decomp: usize,
    /// num_limbs is the number of limbs we decompose each input into, not including the last shifted limb
    pub num_limbs: usize,
}

impl IsLessThanAir {
    pub fn new(bus_index: usize, max_bits: usize, decomp: usize) -> Self {
        Self {
            bus_index,
            max_bits,
            decomp,
            num_limbs: (max_bits + decomp - 1) / decomp,
        }
    }

    /// FOR INTERNAL USE ONLY.
    /// This AIR is only sound if interactions are enabled
    pub(crate) fn eval_without_interactions<AB: AirBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<AB::Expr>,
        aux: IsLessThanAuxCols<AB::Var>,
    ) {
        let x = io.x;
        let y = io.y;
        let less_than = io.less_than;

        let local_aux = &aux;

        let lower = local_aux.lower;
        let lower_decomp = local_aux.lower_decomp.clone();

        // this is the desired intermediate value (i.e. 2^limb_bits + y - x - 1)
        let intermed_val =
            y - x + AB::Expr::from_canonical_u64(1 << self.max_bits) - AB::Expr::one();

        // constrain that the lower_bits + less_than * 2^limb_bits is the correct intermediate sum
        // note that the intermediate value will be >= 2^limb_bits if and only if x < y, and check_val will therefore be
        // the correct value if and only if less_than is the indicator for whether x < y
        let check_val =
            lower + less_than.clone() * AB::Expr::from_canonical_u64(1 << self.max_bits);

        builder.assert_eq(intermed_val, check_val);

        // The following constrains that lower is of at most limb_bits bits

        // constrain that the decomposition of lower_bits is correct
        // each limb will be range checked
        let lower_from_decomp = lower_decomp
            .iter()
            .enumerate()
            .take(self.num_limbs)
            .fold(AB::Expr::zero(), |acc, (i, &val)| {
                acc + val * AB::Expr::from_canonical_u64(1 << (i * self.decomp))
            });

        builder.assert_eq(lower_from_decomp, lower);

        // Ensuring, in case decomp does not divide max_bits, then the last lower_decomp is
        // shifted correctly
        if self.max_bits % self.decomp != 0 {
            let last_limb_shift = (self.decomp - (self.max_bits % self.decomp)) % self.decomp;

            builder.assert_eq(
                (*lower_decomp.last().unwrap()).into(),
                lower_decomp[lower_decomp.len() - 2]
                    * AB::Expr::from_canonical_u64(1 << last_limb_shift),
            );
        }

        // constrain that less_than is a boolean
        builder.assert_bool(less_than);
    }
}

impl AirConfig for IsLessThanAir {
    type Cols<T> = IsLessThanCols<T>;
}

impl<F: Field> BaseAir<F> for IsLessThanAir {
    fn width(&self) -> usize {
        IsLessThanCols::<F>::width(self)
    }
}

impl<AB: InteractionBuilder> Air<AB> for IsLessThanAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanCols::<AB::Var>::from_slice(local);

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-air with constraints to check whether one number is less than another
impl<AB: InteractionBuilder> SubAir<AB> for IsLessThanAir {
    type IoView = IsLessThanIoCols<AB::Var>;
    type AuxView = IsLessThanAuxCols<AB::Var>;

    // constrain that the result of x < y is given by less_than
    // warning: send for range check must be included for the constraints to be sound
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        // Note: every AIR that uses this sub-AIR must include the interactions for soundness
        self.subair_eval(
            builder,
            IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than),
            aux,
        );
    }
}

impl IsLessThanAir {
    pub fn subair_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        io: IsLessThanIoCols<AB::Expr>,
        aux: IsLessThanAuxCols<AB::Var>,
    ) {
        let io_exprs = IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than);

        self.eval_interactions(builder, aux.lower_decomp.clone());
        self.eval_without_interactions(builder, io_exprs, aux);
    }

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
        io: IsLessThanIoCols<impl Into<AB::Expr>>,
        aux: IsLessThanAuxCols<AB::Var>,
    ) {
        let io_exprs = IsLessThanIoCols::<AB::Expr>::new(io.x, io.y, io.less_than);

        self.eval_interactions(builder, aux.lower_decomp.clone());
        self.eval_without_interactions(&mut builder.when_transition(), io_exprs, aux);
    }
}
