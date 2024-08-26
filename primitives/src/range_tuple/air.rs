use afs_stark_backend::interaction::InteractionBuilder;
use p3_air::{Air, BaseAir, PairBuilder};
use p3_field::Field;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::columns::{RangeTupleCols, RangeTuplePreprocessedCols};

#[derive(Clone, Default, Debug)]
pub struct RangeTupleCheckerAir {
    pub bus_index: usize,
    pub sizes: Vec<u32>,
}

impl<F: Field> BaseAir<F> for RangeTupleCheckerAir {
    fn width(&self) -> usize {
        self.sizes.len()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let height: u32 = self.sizes.iter().product();
        let mut unrolled_matrix = Vec::with_capacity((height as usize) * self.sizes.len());
        let mut row = vec![0u32; self.sizes.len()];
        for _ in 0..height {
            unrolled_matrix.extend(row.clone());
            for i in (0..self.sizes.len()).rev() {
                if row[i] < self.sizes[i] - 1 {
                    row[i] += 1;
                    break;
                }
                row[i] = 0;
            }
        }
        Some(RowMajorMatrix::new(
            unrolled_matrix
                .iter()
                .map(|&v| F::from_canonical_u32(v))
                .collect(),
            self.sizes.len(),
        ))
    }
}

impl<AB: InteractionBuilder + PairBuilder> Air<AB> for RangeTupleCheckerAir {
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_local = preprocessed.row_slice(0);
        let prep_local = RangeTuplePreprocessedCols {
            counters: (*prep_local).to_vec(),
        };
        let main = builder.main();
        let local = main.row_slice(0);
        let local = RangeTupleCols { mult: (*local)[0] };
        self.eval_interactions(
            builder,
            prep_local
                .counters
                .iter()
                .map(|&v| v.into())
                .collect::<Vec<_>>(),
            local.mult,
        );
    }
}
