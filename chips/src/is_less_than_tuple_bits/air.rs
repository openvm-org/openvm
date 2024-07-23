use std::borrow::Borrow;

use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::{
    is_equal::{
        columns::{IsEqualAuxCols, IsEqualCols, IsEqualIoCols},
        IsEqualAir,
    },
    is_less_than_bits::{
        columns::{IsLessThanBitsCols, IsLessThanBitsIoCols},
        IsLessThanBitsAir,
    },
    sub_chip::{AirConfig, SubAir},
};

use super::columns::{
    IsLessThanTupleBitsAuxCols, IsLessThanTupleBitsCols, IsLessThanTupleBitsIoCols,
};

#[derive(Clone, Debug)]
pub struct IsLessThanTupleBitsAir {
    // IsLessThanAirs for each tuple element
    pub is_less_than_bits_airs: Vec<IsLessThanBitsAir>,
}

impl IsLessThanTupleBitsAir {
    pub fn new(limb_bits: Vec<usize>) -> Self {
        let is_less_than_bits_airs = limb_bits
            .iter()
            .map(|&limb_bit| IsLessThanBitsAir::new(limb_bit))
            .collect::<Vec<_>>();

        Self {
            is_less_than_bits_airs,
        }
    }

    pub fn tuple_len(&self) -> usize {
        self.is_less_than_bits_airs.len()
    }

    pub fn limb_bits(&self) -> Vec<usize> {
        self.is_less_than_bits_airs
            .iter()
            .map(|air| air.limb_bits)
            .collect()
    }
}

impl AirConfig for IsLessThanTupleBitsAir {
    type Cols<T> = IsLessThanTupleBitsCols<T>;
}

impl<F: Field> BaseAir<F> for IsLessThanTupleBitsAir {
    fn width(&self) -> usize {
        IsLessThanTupleBitsCols::<F>::get_width(self.limb_bits().clone(), self.tuple_len())
    }
}

impl<AB: AirBuilder> Air<AB> for IsLessThanTupleBitsAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &[AB::Var] = (*local).borrow();

        let local_cols = IsLessThanTupleBitsCols::<AB::Var>::from_slice(
            local,
            self.limb_bits().clone(),
            self.tuple_len(),
        );

        SubAir::eval(self, builder, local_cols.io, local_cols.aux);
    }
}

// sub-chip with constraints to check whether one tuple is less than the another
impl<AB: AirBuilder> SubAir<AB> for IsLessThanTupleBitsAir {
    type IoView = IsLessThanTupleBitsIoCols<AB::Var>;
    type AuxView = IsLessThanTupleBitsAuxCols<AB::Var>;

    // constrain that x < y lexicographically
    fn eval(&self, builder: &mut AB, io: Self::IoView, aux: Self::AuxView) {
        let x = io.x.clone();
        let y = io.y.clone();

        // here we constrain that less_than[i] indicates whether x[i] < y[i] using the IsLessThan subchip for each i
        for i in 0..x.len() {
            let x_val = x[i];
            let y_val = y[i];

            let is_less_than_cols = IsLessThanBitsCols {
                io: IsLessThanBitsIoCols {
                    x: x_val,
                    y: y_val,
                    is_less_than: aux.less_than[i],
                },
                aux: aux.less_than_aux[i].clone(),
            };

            SubAir::eval(
                &self.is_less_than_bits_airs[i].clone(),
                builder,
                is_less_than_cols.io,
                is_less_than_cols.aux,
            );
        }

        // here, we constrain that is_equal is the indicator for whether diff == 0, i.e. x[i] = y[i]
        for i in 0..x.len() {
            let is_equal = aux.is_equal[i];
            let inv = aux.is_equal_aux[i].inv;

            let is_equal_cols = IsEqualCols {
                io: IsEqualIoCols {
                    x: x[i],
                    y: y[i],
                    is_equal,
                },
                aux: IsEqualAuxCols { inv },
            };

            SubAir::eval(&IsEqualAir, builder, is_equal_cols.io, is_equal_cols.aux);
        }

        let less_than_cumulative = aux.less_than_cumulative.clone();
        builder.assert_eq(less_than_cumulative[0], aux.less_than[0]);
        for i in 1..x.len() {
            builder.assert_eq(
                less_than_cumulative[i],
                aux.less_than[i] + (aux.is_equal[i] * less_than_cumulative[i - 1]),
            );
        }

        // constrain that the tuple_less_than does indicate whether x < y, lexicographically
        builder.assert_eq(io.tuple_less_than, less_than_cumulative[x.len() - 1]);
    }
}
