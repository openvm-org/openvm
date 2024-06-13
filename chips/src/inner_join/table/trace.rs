use p3_field::PrimeField;
use p3_matrix::dense::RowMajorMatrix;

use super::TableAir;

impl TableAir {
    pub fn gen_table_trace<F: PrimeField>(&self, table: Vec<Vec<u32>>) -> RowMajorMatrix<F> {
        assert!(!table.is_empty());
        let width = table[0].len();

        RowMajorMatrix::new(
            table
                .into_iter()
                .flat_map(|row| row.into_iter().map(F::from_canonical_u32))
                .collect(),
            width,
        )
    }
}
