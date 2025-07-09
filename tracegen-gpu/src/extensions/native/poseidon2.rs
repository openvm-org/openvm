use std::{slice::from_raw_parts, sync::Arc};

use openvm_circuit::{arch::DenseRecordArena, utils::next_power_of_two_or_zero};
use openvm_native_circuit::air::NativePoseidon2Air;
use openvm_stark_backend::{
    p3_field::FieldAlgebra, p3_matrix::dense::RowMajorMatrix, rap::get_air_name, AirRef,
    ChipUsageGetter,
};
use p3_air::BaseAir;
use stark_backend_gpu::{
    base::DeviceMatrix,
    data_transporter::transport_matrix_to_device,
    prover_backend::GpuBackend,
    types::{F, SC},
};

use crate::{
    extensions::native::poseidon2_cuda, primitives::var_range::VariableRangeCheckerChipGPU,
    DeviceChip,
};

pub struct NativePoseidon2ChipGpu<const SBOX_REGISTERS: usize> {
    pub air: NativePoseidon2Air<F, SBOX_REGISTERS>,
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub arena: DenseRecordArena,
}

impl<const SBOX_REGISTERS: usize> NativePoseidon2ChipGpu<SBOX_REGISTERS> {
    pub fn new(
        air: NativePoseidon2Air<F, SBOX_REGISTERS>,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        arena: DenseRecordArena,
    ) -> Self {
        Self {
            air,
            range_checker,
            arena,
        }
    }
}

impl<const SBOX_REGISTERS: usize> ChipUsageGetter for NativePoseidon2ChipGpu<SBOX_REGISTERS> {
    fn air_name(&self) -> String {
        get_air_name(&self.air)
    }

    fn current_trace_height(&self) -> usize {
        let record_size = self.trace_width() * size_of::<F>();
        let records_len = self.arena.allocated().len();
        assert_eq!(records_len % record_size, 0);
        records_len / record_size
    }

    fn trace_width(&self) -> usize {
        BaseAir::<F>::width(&self.air)
    }
}

impl<const SBOX_REGISTERS: usize> DeviceChip<SC, GpuBackend>
    for NativePoseidon2ChipGpu<SBOX_REGISTERS>
{
    fn air(&self) -> AirRef<SC> {
        Arc::new(self.air.clone())
    }

    fn generate_trace(&self) -> DeviceMatrix<F> {
        let num_records = self.current_trace_height();
        let height = next_power_of_two_or_zero(num_records);
        let width = self.trace_width();

        let mut h_trace = vec![F::ZERO; height * width];
        unsafe {
            let bytes = self.arena.allocated();
            assert_eq!(bytes.len() % size_of::<F>(), 0);
            assert_eq!(bytes.as_ptr() as usize % std::mem::align_of::<F>(), 0);
            let slice = from_raw_parts(bytes.as_ptr() as *const F, bytes.len() / size_of::<F>());
            h_trace[..slice.len()].copy_from_slice(slice);
        }

        let d_trace = transport_matrix_to_device(Arc::new(RowMajorMatrix::new(h_trace, width)));
        unsafe {
            poseidon2_cuda::inplace_tracegen(
                &d_trace,
                num_records,
                self.range_checker.count.as_ref(),
                SBOX_REGISTERS,
            )
            .expect("Failed to generate trace");
        }
        d_trace
    }
}

#[cfg(test)]
mod tests {
    use std::{array::from_fn, cmp::min};

    use openvm_circuit::{
        arch::{testing::memory::gen_pointer, NewVmChipWrapper},
        system::memory::SharedMemoryHelper,
    };
    use openvm_instructions::{instruction::Instruction, LocalOpcode};
    use openvm_native_circuit::{
        air::VerifyBatchBus,
        chip::{NativePoseidon2RecordMut, NativePoseidon2Step},
        new_native_poseidon2_chip, NativePoseidon2Chip,
    };
    use openvm_native_compiler::{conversion::AS, Poseidon2Opcode, VerifyBatchOpcode};
    use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
    use openvm_stark_backend::{
        p3_field::{Field, FieldAlgebra, PrimeField32},
        verifier::VerificationError,
    };
    use openvm_stark_sdk::utils::create_seeded_rng;
    use rand::{rngs::StdRng, Rng};
    use test_case::test_case;

    use super::*;
    use crate::testing::GpuChipTestBuilder;

    const MAX_INS_CAPACITY: usize = 128;
    const SBOX_REGISTERS: usize = 1;
    const CHUNK: usize = 8;

    fn create_sparse_chip(
        tester: &GpuChipTestBuilder,
        config: Poseidon2Config<F>,
    ) -> NativePoseidon2Chip<F, SBOX_REGISTERS> {
        let mut chip = new_native_poseidon2_chip(
            tester.system_port(),
            config,
            VerifyBatchBus::new(7),
            tester.cpu_memory_helper(),
        );
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    fn create_dense_chip(
        air: NativePoseidon2Air<F, SBOX_REGISTERS>,
        step: NativePoseidon2Step<F, SBOX_REGISTERS>,
        mem_helper: SharedMemoryHelper<F>,
    ) -> NewVmChipWrapper<
        F,
        NativePoseidon2Air<F, SBOX_REGISTERS>,
        NativePoseidon2Step<F, SBOX_REGISTERS>,
        DenseRecordArena,
    > {
        let mut chip = NewVmChipWrapper::<
            F,
            NativePoseidon2Air<F, SBOX_REGISTERS>,
            NativePoseidon2Step<F, SBOX_REGISTERS>,
            DenseRecordArena,
        >::new(air, step, mem_helper);
        chip.set_trace_buffer_height(MAX_INS_CAPACITY);
        chip
    }

    #[test_case(Poseidon2Opcode::PERM_POS2)]
    #[test_case(Poseidon2Opcode::COMP_POS2)]
    fn test_poseidon2_chip_gpu(opcode: Poseidon2Opcode) {
        let mut rng = create_seeded_rng();
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();

        let mut sparse_chip = create_sparse_chip(&tester, Poseidon2Config::default());
        let mut dense_chip = create_dense_chip(
            sparse_chip.air.clone(),
            NativePoseidon2Step::new(Poseidon2Config::default()),
            tester.cpu_memory_helper(),
        );

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

            tester.write(d, a, [F::from_canonical_usize(dst)]);
            tester.write(d, b, [F::from_canonical_usize(lhs)]);
            if opcode == Poseidon2Opcode::COMP_POS2 {
                tester.write(d, c, [F::from_canonical_usize(rhs)]);
            }

            let data_left: [_; CHUNK] =
                from_fn(|_| F::from_canonical_usize(rng.gen_range(1..=100)));
            let data_right: [_; CHUNK] =
                from_fn(|_| F::from_canonical_usize(rng.gen_range(1..=100)));
            match opcode {
                Poseidon2Opcode::COMP_POS2 => {
                    tester.write(e, lhs, data_left);
                    tester.write(e, rhs, data_right);
                }
                Poseidon2Opcode::PERM_POS2 => {
                    tester.write(e, lhs, data_left);
                    tester.write(e, lhs + CHUNK, data_right);
                }
            }

            tester.execute(&mut dense_chip, &instruction);
        }

        type Record<'a> = NativePoseidon2RecordMut<'a, F, SBOX_REGISTERS>;
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut sparse_chip.arena);

        let gpu_chip =
            NativePoseidon2ChipGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
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
        let mut tester = GpuChipTestBuilder::default().with_variable_range_checker();
        const ADDRESS_SPACE: usize = AS::Native as usize;

        let config = Poseidon2Config::default();
        let hasher = Poseidon2SubChip::<F, SBOX_REGISTERS>::new(config.constants);

        let mut sparse_chip = create_sparse_chip(&tester, config);
        let mut dense_chip = create_dense_chip(
            sparse_chip.air.clone(),
            NativePoseidon2Step::new(config),
            tester.cpu_memory_helper(),
        );

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
                &mut dense_chip,
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
        dense_chip
            .arena
            .get_record_seeker::<Record, _>()
            .transfer_to_matrix_arena(&mut sparse_chip.arena);

        let gpu_chip =
            NativePoseidon2ChipGpu::new(dense_chip.air, tester.range_checker(), dense_chip.arena);

        tester
            .build()
            .load_and_compare(gpu_chip, sparse_chip)
            .finalize()
            .simple_test_with_expected_error(VerificationError::ChallengePhaseError);
    }
}
