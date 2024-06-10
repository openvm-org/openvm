use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::dense::RowMajorMatrix;

use super::columns::NUM_XOR_LOOKUP_COLS;
use super::XorLookupAir;

impl<F: Field, const M: usize> BaseAir<F> for XorLookupAir<M> {
    fn width(&self) -> usize {
        NUM_XOR_LOOKUP_COLS
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows: Vec<Vec<F>> = (0..(1 << M) * (1 << M))
            .map(|i| {
                let x = i / (1 << M);
                let y = i % (1 << M);
                let z = x ^ y;
                vec![
                    F::from_canonical_usize(x),
                    F::from_canonical_usize(y),
                    F::from_canonical_usize(z),
                ]
            })
            .collect();

        Some(RowMajorMatrix::new(rows.concat(), 3))
    }
}

impl<AB, const M: usize> Air<AB> for XorLookupAir<M>
where
    AB: AirBuilder,
{
    fn eval(&self, _builder: &mut AB) {}
}
