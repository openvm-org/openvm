use p3_field::PrimeField32;
use p3_keccak_air::{generate_trace_rows, KeccakCols, NUM_KECCAK_COLS, NUM_ROUNDS, U64_LIMBS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use tiny_keccak::keccakf;

use super::{columns::NUM_KECCAK_PERMUTE_COLS, KeccakPermuteChip, KeccakVmChip};
use crate::{
    cpu::{trace::Instruction, OpCode},
    vm::ExecutionSegment,
};

impl<F: PrimeField32> KeccakVmChip<F> {
    pub fn generate_trace(&mut self) -> RowMajorMatrix<F> {
        let mut requests = std::mem::take(&mut self.requests);
        let inputs = std::mem::take(&mut self.inputs);
        assert_eq!(requests.len(), inputs.len());
        let p3_keccak_trace: RowMajorMatrix<F> = generate_trace_rows(inputs);
        let num_rows = p3_keccak_trace.height();
        // Every `NUM_ROUNDS` rows corresponds to one opcode call
        let num_opcode_calls = (num_rows + NUM_ROUNDS - 1) / NUM_ROUNDS;
        // Resize with dummy `is_opcode = 0`
        requests.resize(num_opcode_calls, Default::default());

        // Use unsafe alignment so we can parallely write to the matrix
        let mut trace = RowMajorMatrix::new(
            vec![F::zero(); num_rows * NUM_KECCAK_PERMUTE_COLS],
            NUM_KECCAK_PERMUTE_COLS,
        );
        let (prefix, rows, suffix) = unsafe { trace.values.align_to_mut::<KeccakPermuteCols<F>>() };
        assert!(prefix.is_empty(), "Alignment should match");
        assert!(suffix.is_empty(), "Alignment should match");
        assert_eq!(rows.len(), num_rows);

        rows.par_chunks_mut(NUM_ROUNDS)
            .zip(
                p3_keccak_trace
                    .values
                    .par_chunks(NUM_KECCAK_COLS * NUM_ROUNDS),
            )
            .zip(requests.into_par_iter())
            .for_each(|((rows, p3_keccak_mat), (io, aux))| {
                for (row, p3_keccak_row) in rows
                    .iter_mut()
                    .zip(p3_keccak_mat.chunks_exact(NUM_KECCAK_COLS))
                {
                    // Cast &mut KeccakCols<F> to &mut [F]:
                    let inner_raw_ptr: *mut KeccakCols<F> = &mut row.inner as *mut _;
                    let row_slice = unsafe {
                        std::slice::from_raw_parts_mut(inner_raw_ptr as *mut F, NUM_KECCAK_COLS)
                    };
                    row_slice.copy_from_slice(p3_keccak_row);
                    row.io = io;
                    row.aux = aux;
                }
            });

        trace
    }
}
