use std::{borrow::Borrow, mem::size_of, slice::from_raw_parts, sync::Arc};

use derive_new::new;
use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_circuit_primitives::var_range::cuda::VariableRangeCheckerChipGPU;
use openvm_cuda_backend::{
    base::DeviceMatrix, chip::get_empty_air_proving_ctx, prover_backend::GpuBackend, types::F,
};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::{prover::types::AirProvingContext, Chip};
use p3_field::{Field, PrimeField32};

use super::columns::NativePoseidon2Cols;
use crate::cuda_abi::poseidon2_cuda;

#[derive(new)]
pub struct NativePoseidon2ChipGpu<const SBOX_REGISTERS: usize> {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl<const SBOX_REGISTERS: usize> Chip<DenseRecordArena, GpuBackend>
    for NativePoseidon2ChipGpu<SBOX_REGISTERS>
{
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        let records = arena.allocated();
        if records.is_empty() {
            return get_empty_air_proving_ctx::<GpuBackend>();
        }

        // For Poseidon2, the records are already the trace rows
        // Use the columns width directly
        let width = NativePoseidon2Cols::<F, SBOX_REGISTERS>::width();

        let record_size = width * size_of::<F>();
        assert_eq!(records.len() % record_size, 0);

        let height = records.len() / record_size;
        let padded_height = next_power_of_two_or_zero(height);

        let d_chunk_start = {
            let mut row_idx = 0;
            let row_slice = unsafe {
                let raw_ptr = records.as_ptr();
                from_raw_parts(raw_ptr as *const F, records.len() / size_of::<F>())
            };
            let mut chunk_start = Vec::new();
            // Allocated rows are not empty. Determine the chunk start indices.
            while row_idx < height {
                let start = row_idx * width;
                let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                    row_slice[start..(start + width)].borrow();
                chunk_start.push(row_idx as u32);
                if cols.simple.is_one() {
                    row_idx += 1;
                } else {
                    let num_non_inside_row = cols.inner.export.as_canonical_u32() as usize;
                    let non_inside_start = start + (num_non_inside_row - 1) * width;
                    let cols: &NativePoseidon2Cols<F, SBOX_REGISTERS> =
                        row_slice[non_inside_start..(non_inside_start + width)].borrow();
                    let total_num_row = cols.inner.export.as_canonical_u32() as usize;
                    row_idx += total_num_row;
                };
            }
            chunk_start.to_device().unwrap()
        };

        let trace = DeviceMatrix::<F>::with_capacity(padded_height, width);

        let d_records = records.to_device().unwrap();

        unsafe {
            poseidon2_cuda::tracegen(
                trace.buffer(),
                padded_height as u32,
                width as u32,
                &d_records,
                height as u32,
                &d_chunk_start,
                d_chunk_start.len() as u32,
                &self.range_checker.count,
                SBOX_REGISTERS as u32,
                self.timestamp_max_bits as u32,
            )
            .unwrap();
        }

        AirProvingContext::simple_no_pis(trace)
    }
}

#[cfg(test)]
mod tests {
    use std::{array::from_fn, cmp::min};

    use crate::poseidon2::{
        air::{NativePoseidon2Air, VerifyBatchBus},
        chip::{NativePoseidon2Executor, NativePoseidon2Filler, NativePoseidon2RecordMut},
        NativePoseidon2Chip,
    };
    use openvm_circuit::arch::testing::memory::gen_pointer;
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_compiler::{conversion::AS, Poseidon2Opcode, VerifyBatchOpcode};
    use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
    use openvm_stark_backend::p3_field::{Field, FieldAlgebra, PrimeField32};
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;

    const MAX_INS_CAPACITY: usize = 128;
    const SBOX_REGISTERS: usize = 1;
    const CHUNK: usize = 8;

    fn create_test_harness(
        tester: &GpuChipTestBuilder,
        config: Poseidon2Config<F>,
    ) -> GpuTestChipHarness<
        F,
        NativePoseidon2Executor<F, SBOX_REGISTERS>,
        NativePoseidon2Air<F, SBOX_REGISTERS>,
        NativePoseidon2ChipGpu<SBOX_REGISTERS>,
        NativePoseidon2Chip<F, SBOX_REGISTERS>,
    > {
        let air = NativePoseidon2Air::new(
            tester.execution_bridge(),
            tester.memory_bridge(),
            VerifyBatchBus::new(7),
            config,
        );
        let executor = NativePoseidon2Executor::new(config);

        let cpu_chip = NativePoseidon2Chip::new(
            NativePoseidon2Filler::new(config),
            tester.dummy_memory_helper(),
        );

        let gpu_chip =
            NativePoseidon2ChipGpu::new(tester.range_checker(), tester.timestamp_max_bits());

        GpuTestChipHarness::with_capacity(executor, air, gpu_chip, cpu_chip, MAX_INS_CAPACITY)
    }

    #[test_case(Poseidon2Opcode::PERM_POS2)]
    #[test_case(Poseidon2Opcode::COMP_POS2)]
    fn test_poseidon2_chip_gpu(opcode: Poseidon2Opcode) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();

        let mut harness = create_test_harness(&tester, Poseidon2Config::default());

        for _ in 0..100 {
            let instruction = Instruction {
                opcode: opcode.global_opcode(),
                a: F::from_canonical_usize(gen_pointer(&mut rng, 1)),
                b: F::from_canonical_usize(gen_pointer(&mut rng, 1)),
                c: F::from_canonical_usize(gen_pointer(&mut rng, 1)),
                d: F::from_canonical_usize(4),
                e: F::from_canonical_usize(4),
                f: F::ZERO,
                g: F::ZERO,
            };

            let dst = gen_pointer(&mut rng, CHUNK) / 2;
            let lhs = gen_pointer(&mut rng, CHUNK) / 2;
            let rhs = gen_pointer(&mut rng, CHUNK) / 2;

            let [a, b, c, d, e] = [
                instruction.a,
                instruction.b,
                instruction.c,
                instruction.d,
                instruction.e,
            ]
            .map(|elem| elem.as_canonical_u32() as usize);

            tester.write::<1>(d, a, [F::from_canonical_usize(dst)]);
            tester.write::<1>(d, b, [F::from_canonical_usize(lhs)]);
            if opcode == Poseidon2Opcode::COMP_POS2 {
                tester.write::<1>(d, c, [F::from_canonical_usize(rhs)]);
            }

            let data_left: [_; CHUNK] =
                from_fn(|_| F::from_canonical_usize(rng.gen_range(1..=100)));
            let data_right: [_; CHUNK] =
                from_fn(|_| F::from_canonical_usize(rng.gen_range(1..=100)));
            match opcode {
                Poseidon2Opcode::COMP_POS2 => {
                    tester.write::<CHUNK>(e, lhs, data_left);
                    tester.write::<CHUNK>(e, rhs, data_right);
                }
                Poseidon2Opcode::PERM_POS2 => {
                    tester.write::<CHUNK>(e, lhs, data_left);
                    tester.write::<CHUNK>(e, lhs + CHUNK, data_right);
                }
            }

            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &instruction,
            );
        }

        type Record<'a> = NativePoseidon2RecordMut<'a, F, SBOX_REGISTERS>;
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }

    #[derive(Debug, Clone)]
    struct VerifyBatchInstance {
        dim: Vec<usize>,
        opened: Vec<Vec<F>>,
        proof: Vec<[F; CHUNK]>,
        sibling_is_on_right: Vec<bool>,
        commit: [F; CHUNK],
    }

    fn compute_commit(
        dim: &[usize],
        opened: &[Vec<F>],
        proof: &[[F; CHUNK]],
        sibling_is_on_right: &[bool],
        hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
    ) -> [F; CHUNK] {
        let mut log_height = dim[0] as isize;
        let mut proof_index = 0;
        let mut opened_index = 0;
        let mut root = [F::ZERO; CHUNK];
        while log_height >= 0 {
            let mut concat = vec![];
            while opened_index < opened.len() && dim[opened_index] == log_height as usize {
                concat.extend(opened[opened_index].clone());
                opened_index += 1;
            }
            if !concat.is_empty() {
                let mut left = [F::ZERO; CHUNK];
                let mut right = [F::ZERO; CHUNK];
                for i in (0..concat.len()).step_by(CHUNK) {
                    left[..(min(i + CHUNK, concat.len()) - i)]
                        .copy_from_slice(&concat[i..min(i + CHUNK, concat.len())]);
                    (left, right) = hash_function(left, right);
                }
                root = if log_height as usize == dim[0] {
                    left
                } else {
                    hash_function(root, left).0
                }
            }
            if log_height > 0 {
                let sibling = proof[proof_index];
                let (left, right) = if sibling_is_on_right[proof_index] {
                    (sibling, root)
                } else {
                    (root, sibling)
                };
                root = hash_function(left, right).0;
            }
            log_height -= 1;
            proof_index += 1;
        }
        root
    }

    fn random_instance(
        rng: &mut StdRng,
        row_lengths: Vec<Vec<usize>>,
        opened_element_size: usize,
        hash_function: impl Fn([F; CHUNK], [F; CHUNK]) -> ([F; CHUNK], [F; CHUNK]),
    ) -> VerifyBatchInstance {
        let mut dims = vec![];
        let mut opened = vec![];
        let mut proof = vec![];
        let mut sibling_is_on_right = vec![];
        for (log_height, row_lengths) in row_lengths.iter().enumerate() {
            for &row_length in row_lengths {
                dims.push(log_height);
                let mut opened_row = vec![];
                for _ in 0..opened_element_size * row_length {
                    opened_row.push(rng.gen());
                }
                opened.push(opened_row);
            }
            if log_height > 0 {
                proof.push(std::array::from_fn(|_| rng.gen()));
                sibling_is_on_right.push(rng.gen());
            }
        }

        dims.reverse();
        opened.reverse();
        proof.reverse();
        sibling_is_on_right.reverse();

        let commit = compute_commit(&dims, &opened, &proof, &sibling_is_on_right, hash_function);

        VerifyBatchInstance {
            dim: dims,
            opened,
            proof,
            sibling_is_on_right,
            commit,
        }
    }

    #[test]
    fn test_verify_batch() {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default();
        const ADDRESS_SPACE: usize = AS::Native as usize;

        let config = Poseidon2Config::default();
        let hasher = Poseidon2SubChip::<F, SBOX_REGISTERS>::new(config.constants);

        let mut harness = create_test_harness(&tester, config);

        let cases: [(Vec<Vec<usize>>, usize); 5] = [
            (vec![vec![3], vec![], vec![9, 2, 1, 13, 4], vec![16]], 1),
            (vec![vec![1, 1, 1], vec![3], vec![2]], 4),
            (vec![vec![8], vec![7], vec![6]], 1),
            (vec![vec![], vec![], vec![], vec![1]], 4),
            (vec![vec![4], vec![3], vec![2]], 4),
        ];

        for (row_lengths, opened_element_size) in cases {
            let instance =
                random_instance(&mut rng, row_lengths, opened_element_size, |left, right| {
                    let concatenated =
                        std::array::from_fn(|i| if i < CHUNK { left[i] } else { right[i - CHUNK] });
                    let permuted = hasher.permute(concatenated);
                    (
                        std::array::from_fn(|i| permuted[i]),
                        std::array::from_fn(|i| permuted[i + CHUNK]),
                    )
                });

            let VerifyBatchInstance {
                dim,
                opened,
                proof,
                sibling_is_on_right,
                commit,
            } = instance;

            let dim_register = gen_pointer(&mut rng, 1);
            let opened_register = gen_pointer(&mut rng, 1);
            let opened_length_register = gen_pointer(&mut rng, 1);
            let proof_id = gen_pointer(&mut rng, 1);
            let index_register = gen_pointer(&mut rng, 1);
            let commit_register = gen_pointer(&mut rng, 1);

            let dim_base_pointer = gen_pointer(&mut rng, 1);
            let opened_base_pointer = gen_pointer(&mut rng, 2);
            let index_base_pointer = gen_pointer(&mut rng, 1);
            let commit_pointer = gen_pointer(&mut rng, 1);

            tester.write_usize(ADDRESS_SPACE, dim_register, [dim_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, opened_register, [opened_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, opened_length_register, [opened.len()]);
            tester.write_usize(ADDRESS_SPACE, proof_id, [tester.streams.hint_space.len()]);
            tester.write_usize(ADDRESS_SPACE, index_register, [index_base_pointer]);
            tester.write_usize(ADDRESS_SPACE, commit_register, [commit_pointer]);

            for (i, &dim_value) in dim.iter().enumerate() {
                tester.write_usize(ADDRESS_SPACE, dim_base_pointer + i, [dim_value]);
            }
            for (i, opened_row) in opened.iter().enumerate() {
                let row_pointer = gen_pointer(&mut rng, 1);
                tester.write_usize(
                    ADDRESS_SPACE,
                    opened_base_pointer + (2 * i),
                    [row_pointer, opened_row.len() / opened_element_size],
                );
                for (j, &opened_value) in opened_row.iter().enumerate() {
                    tester.write(ADDRESS_SPACE, row_pointer + j, [opened_value]);
                }
            }

            tester
                .streams
                .hint_space
                .push(proof.iter().flatten().copied().collect());
            for (i, &bit) in sibling_is_on_right.iter().enumerate() {
                tester.write(ADDRESS_SPACE, index_base_pointer + i, [F::from_bool(bit)]);
            }
            tester.write(ADDRESS_SPACE, commit_pointer, commit);

            let opened_element_size_inv = F::from_canonical_usize(opened_element_size)
                .inverse()
                .as_canonical_u32() as usize;
            tester.execute(
                &mut harness.executor,
                &mut harness.dense_arena,
                &Instruction::from_usize(
                    VerifyBatchOpcode::VERIFY_BATCH.global_opcode(),
                    [
                        dim_register,
                        opened_register,
                        opened_length_register,
                        proof_id,
                        index_register,
                        commit_register,
                        opened_element_size_inv,
                    ],
                ),
            );
        }

        type Record<'a> = NativePoseidon2RecordMut<'a, F, SBOX_REGISTERS>;
        harness
            .dense_arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut harness.matrix_arena);

        tester
            .build()
            .load_gpu_harness(harness)
            .finalize()
            .simple_test()
            .unwrap();
    }
}
