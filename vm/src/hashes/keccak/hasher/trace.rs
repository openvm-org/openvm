use p3_field::PrimeField32;
use p3_keccak_air::{generate_trace_rows, KeccakCols, NUM_KECCAK_COLS, NUM_ROUNDS, U64_LIMBS};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;
use tiny_keccak::keccakf;

use super::{columns::NUM_KECCAK_PERMUTE_COLS, KeccakPermuteChip};
use crate::{
    cpu::{trace::Instruction, OpCode},
    hashes::keccak::hasher::columns::{
        KeccakPermuteAuxCols, KeccakPermuteCols, KeccakPermuteIoCols,
    },
    vm::ExecutionSegment,
};

impl<F: PrimeField32> KeccakPermuteChip<F> {
    /// Wrapper function for tiny-keccak's keccak-f permutation.
    /// Returns the new state after permutation.
    pub fn keccak_f(mut input: [u64; 25]) -> [u64; 25] {
        keccakf(&mut input);
        input
    }

    // TODO: only WORD_SIZE=1 works right now
    pub fn keccak_permute<const WORD_SIZE: usize>(
        vm: &mut ExecutionSegment<WORD_SIZE, F>,
        start_timestamp: usize,
        instruction: Instruction<F>,
    ) {
        assert_eq!(WORD_SIZE, 1, "Only WORD_SIZE=1 supported for now");
        let Instruction {
            opcode,
            op_a,
            op_b,
            op_c,
            d,
            e,
            debug: _debug,
        } = instruction;
        debug_assert_eq!(opcode, OpCode::PERM_KECCAK);
        debug_assert_eq!(op_b, F::zero());

        let mut timestamp = start_timestamp;
        let mut read = |address_space, addr| {
            let val = vm.memory_chip.read_elem(timestamp, address_space, addr);
            timestamp += 1;
            val
        };

        let dst = read(d, op_a);
        let src = read(d, op_c);

        let io = KeccakPermuteIoCols::new(
            F::from_bool(true),
            F::from_canonical_usize(start_timestamp),
            op_a,
            op_c,
            d,
            e,
        );
        let aux = KeccakPermuteAuxCols::new(dst, src);

        // TODO: unoptimized, many conversions to/from Montgomery form
        let mut offset = 0;
        let input: [[F; U64_LIMBS]; 25] = [(); 25].map(|_| {
            [(); U64_LIMBS].map(|_| {
                let val = read(e, src + F::from_canonical_usize(offset));
                offset += 1;
                val
            })
        });
        // We need to compute the output to write into memory since runtime is serial
        let input_u64: [u64; 25] = input.map(|limbs| {
            let mut val = 0u64;
            for (i, limb) in limbs.into_iter().enumerate() {
                val |= limb.as_canonical_u64() << (i * 16);
            }
            val
        });
        let output_u64 = Self::keccak_f(input_u64);
        let output: [[F; U64_LIMBS]; 25] = output_u64
            .map(|val| core::array::from_fn(|i| F::from_canonical_u64((val >> (i * 16)) & 0xFFFF)));
        debug_assert_eq!(start_timestamp + Self::write_timestamp_offset(), timestamp);
        // TODO: again very unoptimized
        let mut write = |address_space, addr, val| {
            vm.memory_chip
                .write_elem(timestamp, address_space, addr, val);
            timestamp += 1;
        };
        for (offset, output) in output.into_iter().flatten().enumerate() {
            write(e, dst + F::from_canonical_usize(offset), output);
        }

        // Add the events to chip state for later trace generation usage
        vm.keccak_permute_chip.requests.push((io, aux));
        vm.keccak_permute_chip.inputs.push(input_u64);
    }

    /// The offset from `start_timestamp` when output is written to memory
    fn write_timestamp_offset() -> usize {
        2 + U64_LIMBS * 25
    }

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
