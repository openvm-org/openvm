use p3_field::PrimeField64;
use p3_matrix::dense::RowMajorMatrix;

use crate::page_rw_checker::page_controller::{OpType, Operation};

use super::ExecutionAir;

impl ExecutionAir {
    /// trace_degree is MAX_OPS
    pub fn generate_trace<F: PrimeField64>(
        &self,
        ops: &Vec<Operation>,
        trace_degree: usize,
    ) -> RowMajorMatrix<F> {
        self.generate_trace_testing(ops, trace_degree, 1)
    }

    /// For testing purposes, we want to see that this is still performant when we add spaces everywhere. Spacing = 1 is normal
    pub fn generate_trace_testing<F: PrimeField64>(
        &self,
        ops: &Vec<Operation>,
        trace_degree: usize,
        spacing: usize,
    ) -> RowMajorMatrix<F> {
        assert!(ops.len() * spacing <= trace_degree);
        let mut blank_row = vec![0; self.air_width()];
        let mut rows = vec![];
        for (i, op) in ops.iter().enumerate() {
            rows.extend(vec![blank_row.clone(); spacing - 1]);
            let mut row = vec![];
            row.push(1);
            row.push(i as u32 + 1);
            row.extend(op.idx.clone());
            row.extend(op.data.clone());
            match op.op_type {
                OpType::Read => row.push(0),
                OpType::Write => row.push(1),
            }
            rows.push(row);
            blank_row[1] += 1;
        }
        rows.resize(trace_degree, blank_row.clone());
        println!("ROWS: {:?}", rows);
        let rows: Vec<Vec<F>> = rows
            .iter()
            .map(|row| row.iter().map(|u| F::from_canonical_u32(*u)).collect())
            .collect();
        RowMajorMatrix::new(rows.concat(), self.air_width())
    }
}
