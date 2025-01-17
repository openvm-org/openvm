use std::{borrow::Borrow, sync::Arc};

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{offline_checker::MemoryBridge, MemoryAddress},
};
use openvm_circuit_primitives::utils::not;
use openvm_instructions::Poseidon2Opcode::{COMP_POS2, PERM_POS2};
use openvm_native_compiler::VerifyBatchOpcode::VERIFY_BATCH;
use openvm_poseidon2_air::{Poseidon2SubAir, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS};
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::{InteractionBuilder, InteractionType},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::{
    chip::{NUM_INITIAL_READS, NUM_SIMPLE_ACCESSES},
    verify_batch::{
        columns::{
            InsideRowSpecificCols, SimplePoseidonSpecificCols, TopLevelSpecificCols,
            VerifyBatchCols,
        },
        CHUNK,
    },
};

#[derive(Clone, Debug)]
pub struct VerifyBatchAir<F: Field, const SBOX_REGISTERS: usize> {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub internal_bus: VerifyBatchBus,
    pub(crate) subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub(super) verify_batch_offset: usize,
    pub(super) simple_offset: usize,
    pub(crate) address_space: F,
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for VerifyBatchAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        VerifyBatchCols::<F, SBOX_REGISTERS>::width()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for VerifyBatchAir<F, SBOX_REGISTERS>
{
}

impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for VerifyBatchAir<F, SBOX_REGISTERS>
{
}

impl<AB: InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for VerifyBatchAir<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0);
        let local: &VerifyBatchCols<AB::Var, SBOX_REGISTERS> = (*local).borrow();
        let next = main.row_slice(1);
        let next: &VerifyBatchCols<AB::Var, SBOX_REGISTERS> = (*next).borrow();

        let &VerifyBatchCols {
            inner: _,
            incorporate_row,
            incorporate_sibling,
            inside_row,
            simple,
            end_inside_row,
            end_top_level,
            start_top_level,
            very_first_timestamp,
            start_timestamp,
            opened_element_size_inv,
            initial_opened_index,
            opened_base_pointer,
            is_exhausted,
            specific,
        } = local;

        let left_input = std::array::from_fn::<_, CHUNK, _>(|i| local.inner.inputs[i]);
        let right_input = std::array::from_fn::<_, CHUNK, _>(|i| local.inner.inputs[i + CHUNK]);
        let left_output = std::array::from_fn::<_, CHUNK, _>(|i| {
            local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
        });
        let right_output = std::array::from_fn::<_, CHUNK, _>(|i| {
            local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i + CHUNK]
        });
        let next_left_input = std::array::from_fn::<_, CHUNK, _>(|i| next.inner.inputs[i]);
        let next_right_input = std::array::from_fn::<_, CHUNK, _>(|i| next.inner.inputs[i + CHUNK]);

        builder.assert_bool(incorporate_row);
        builder.assert_bool(incorporate_sibling);
        builder.assert_bool(inside_row);
        builder.assert_bool(simple);
        let enabled = incorporate_row + incorporate_sibling + inside_row + simple;
        builder.assert_bool(enabled.clone());
        builder.assert_bool(end_inside_row);
        builder.when(end_inside_row).assert_one(inside_row);
        builder.assert_bool(end_top_level);

        let end = end_inside_row + end_top_level + simple + (AB::Expr::ONE - enabled.clone());
        builder
            .when(end.clone())
            .when(next.incorporate_row)
            .assert_one(next.start_top_level);

        // poseidon2 constraints are always checked
        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        //// inside row constraints

        let inside_row_specific: &InsideRowSpecificCols<AB::Var> =
            specific[..InsideRowSpecificCols::<AB::Var>::width()].borrow();
        let cells = inside_row_specific.cells;
        let next_inside_row_specific: &InsideRowSpecificCols<AB::Var> =
            next.specific[..InsideRowSpecificCols::<AB::Var>::width()].borrow();
        let next_cells = next_inside_row_specific.cells;

        // start
        builder
            .when(end.clone())
            .when(next.inside_row)
            .assert_eq(next.initial_opened_index, next_cells[0].opened_index);
        builder
            .when(end.clone())
            .when(next.inside_row)
            .assert_eq(next.very_first_timestamp, next.start_timestamp);

        // end
        self.internal_bus.interact(
            builder,
            false,
            end_inside_row,
            very_first_timestamp,
            start_timestamp + AB::F::from_canonical_usize(2 * CHUNK),
            opened_base_pointer,
            opened_element_size_inv,
            initial_opened_index,
            cells[CHUNK - 1].opened_index,
            left_output,
        );

        // things that stay the same (roughly)

        builder.when(inside_row - end_inside_row).assert_eq(
            next.start_timestamp,
            start_timestamp + AB::F::from_canonical_usize(2 * CHUNK),
        );
        builder
            .when(inside_row - end_inside_row)
            .assert_eq(next.opened_base_pointer, opened_base_pointer);
        builder
            .when(inside_row - end_inside_row)
            .assert_eq(next.opened_element_size_inv, opened_element_size_inv);
        builder
            .when(inside_row - end_inside_row)
            .assert_eq(next.initial_opened_index, initial_opened_index);
        builder
            .when(inside_row - end_inside_row)
            .assert_eq(next.very_first_timestamp, very_first_timestamp);

        // right input

        for &next_right_input in next_right_input.iter() {
            builder
                .when(end.clone())
                .when(next.inside_row)
                .assert_zero(next_right_input);
        }

        for i in 0..CHUNK {
            builder
                .when(inside_row - end_inside_row)
                .assert_eq(right_output[i], next_right_input[i]);
        }

        // left input

        // handle exhausted cells on next row

        for i in 0..CHUNK {
            builder
                .when(inside_row - end_inside_row)
                .when(next.is_exhausted[i])
                .assert_eq(next_left_input[i], left_output[i]);
            builder
                .when(end.clone())
                .when(next.is_exhausted[i])
                .assert_zero(next_left_input[i]);
        }

        for i in 0..CHUNK {
            let cell = cells[i];
            let next_cell = if i + 1 == CHUNK {
                next_cells[0]
            } else {
                cells[i + 1]
            };
            let next_is_exhausted = if i + 1 == CHUNK {
                next.is_exhausted[0]
            } else {
                is_exhausted[i + 1]
            };
            let is_exhausted = is_exhausted[i];

            builder.when(inside_row).assert_bool(cell.is_first_in_row);
            builder.assert_bool(is_exhausted);
            builder
                .when(inside_row)
                .assert_bool(cell.is_first_in_row + is_exhausted);

            let next_is_normal = AB::Expr::ONE - next_cell.is_first_in_row - next_is_exhausted;
            self.memory_bridge
                .read(
                    MemoryAddress::new(self.address_space, cell.row_pointer),
                    [left_input[i]],
                    start_timestamp + AB::F::from_canonical_usize((2 * i) + 1),
                    &cell.read,
                )
                .eval(builder, inside_row * not(is_exhausted));

            let mut when_inside_row_not_last = if i == CHUNK - 1 {
                builder.when(inside_row - end_inside_row)
            } else {
                builder.when(inside_row)
            };
            // everthing above oks

            // update state for normal cell
            when_inside_row_not_last
                .when(next_is_normal.clone())
                .assert_eq(next_cell.row_pointer, cell.row_pointer + AB::F::ONE);
            when_inside_row_not_last
                .when(next_is_normal.clone())
                .assert_eq(next_cell.row_end, cell.row_end);
            when_inside_row_not_last
                .when(AB::Expr::ONE - next_cell.is_first_in_row)
                .assert_eq(next_cell.opened_index, cell.opened_index);

            // update state for first in row cell
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        self.address_space,
                        opened_base_pointer + (cell.opened_index * AB::F::TWO),
                    ),
                    [
                        cell.row_pointer.into(),
                        opened_element_size_inv * (cell.row_end - cell.row_pointer),
                    ],
                    start_timestamp + AB::F::from_canonical_usize(2 * i),
                    &cell.read_row_pointer_and_length,
                )
                .eval(builder, inside_row * cell.is_first_in_row);
            let mut when_inside_row_not_last = if i == CHUNK - 1 {
                builder.when(inside_row - end_inside_row)
            } else {
                builder.when(inside_row)
            };
            when_inside_row_not_last
                .when(next_cell.is_first_in_row)
                .assert_eq(next_cell.opened_index, cell.opened_index + AB::F::ONE);

            when_inside_row_not_last
                .when(next_is_exhausted)
                .assert_eq(next_cell.opened_index, cell.opened_index);

            when_inside_row_not_last
                .when(is_exhausted)
                .assert_eq(next_is_exhausted, AB::F::ONE);

            let is_last_in_row = if i == CHUNK - 1 {
                end_inside_row.into()
            } else {
                next_cell.is_first_in_row + next_is_exhausted
            } - is_exhausted;
            builder
                .when(inside_row)
                .when(is_last_in_row)
                .assert_eq(cell.row_pointer + AB::F::ONE, cell.row_end);
        }

        //// top level constraints

        let top_level_specific: &TopLevelSpecificCols<AB::Var> =
            specific[..TopLevelSpecificCols::<AB::Var>::width()].borrow();
        let &TopLevelSpecificCols {
            pc,
            end_timestamp,
            dim_register,
            opened_register,
            opened_length_register,
            sibling_register,
            index_register,
            commit_register,
            final_opened_index,
            height,
            opened_length,
            dim_base_pointer,
            sibling_base_pointer,
            index_base_pointer,
            dim_base_pointer_read,
            opened_base_pointer_read,
            opened_length_read,
            sibling_base_pointer_read,
            index_base_pointer_read,
            commit_pointer_read,
            proof_index,
            read_initial_height_or_root_is_on_right,
            read_final_height_or_sibling_array_start,
            root_is_on_right,
            sibling_array_start,
            reads,
            commit_pointer,
            commit_read,
        } = top_level_specific;
        let next_top_level_specific: &TopLevelSpecificCols<AB::Var> =
            next.specific[..TopLevelSpecificCols::<AB::Var>::width()].borrow();

        builder
            .when(end.clone())
            .when(next.incorporate_row + next.incorporate_sibling)
            .assert_eq(next_top_level_specific.proof_index, AB::F::ZERO);

        let timestamp_after_initial_reads =
            start_timestamp + AB::F::from_canonical_usize(NUM_INITIAL_READS);

        builder
            .when(end.clone())
            .when(next.incorporate_row)
            .assert_eq(next.initial_opened_index, AB::F::ZERO);
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(VERIFY_BATCH as usize + self.verify_batch_offset),
                [
                    dim_register,
                    opened_register,
                    opened_length_register,
                    sibling_register,
                    index_register,
                    commit_register,
                    opened_element_size_inv,
                ],
                ExecutionState::new(pc, very_first_timestamp),
                end_timestamp - very_first_timestamp,
            )
            .eval(builder, end_top_level);

        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, dim_register),
                [dim_base_pointer],
                very_first_timestamp,
                &dim_base_pointer_read,
            )
            .eval(builder, end_top_level);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, opened_register),
                [opened_base_pointer],
                very_first_timestamp + AB::F::ONE,
                &opened_base_pointer_read,
            )
            .eval(builder, end_top_level);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, opened_length_register),
                [opened_length],
                very_first_timestamp + AB::F::TWO,
                &opened_length_read,
            )
            .eval(builder, end_top_level);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, sibling_register),
                [sibling_base_pointer],
                very_first_timestamp + AB::F::from_canonical_usize(3),
                &sibling_base_pointer_read,
            )
            .eval(builder, end_top_level);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, index_register),
                [index_base_pointer],
                very_first_timestamp + AB::F::from_canonical_usize(4),
                &index_base_pointer_read,
            )
            .eval(builder, end_top_level);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, commit_register),
                [commit_pointer],
                very_first_timestamp + AB::F::from_canonical_usize(5),
                &commit_pointer_read,
            )
            .eval(builder, end_top_level);

        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, commit_pointer),
                left_output,
                very_first_timestamp + AB::F::from_canonical_usize(6),
                &commit_read,
            )
            .eval(builder, end_top_level);

        let mut when_top_level_not_end =
            builder.when(incorporate_row + incorporate_sibling - end_top_level);

        when_top_level_not_end
            .assert_eq(next_top_level_specific.dim_base_pointer, dim_base_pointer);

        when_top_level_not_end.assert_eq(next.opened_base_pointer, opened_base_pointer);
        when_top_level_not_end.assert_eq(
            next_top_level_specific.sibling_base_pointer,
            sibling_base_pointer,
        );
        when_top_level_not_end.assert_eq(
            next_top_level_specific.index_base_pointer,
            index_base_pointer,
        );
        when_top_level_not_end.assert_eq(next.start_timestamp, end_timestamp);
        when_top_level_not_end.assert_eq(next_top_level_specific.opened_length, opened_length);
        when_top_level_not_end.assert_eq(next.opened_element_size_inv, opened_element_size_inv);
        when_top_level_not_end
            .assert_eq(next.initial_opened_index, final_opened_index + AB::F::ONE);

        builder
            .when(incorporate_sibling)
            .when(AB::Expr::ONE - end_top_level)
            .assert_eq(next_top_level_specific.height * AB::F::TWO, height);
        builder
            .when(incorporate_row)
            .when(AB::Expr::ONE - end_top_level)
            .assert_eq(next_top_level_specific.height, height);
        builder
            .when(incorporate_sibling)
            .when(AB::Expr::ONE - end_top_level)
            .assert_eq(
                next_top_level_specific.proof_index,
                proof_index + AB::F::ONE,
            );
        builder
            .when(incorporate_row)
            .when(AB::Expr::ONE - end_top_level)
            .assert_eq(next_top_level_specific.proof_index, proof_index);

        builder
            .when(end_top_level)
            .when(incorporate_row)
            .assert_eq(height, AB::F::ONE);
        builder
            .when(end_top_level)
            .when(incorporate_sibling)
            .assert_eq(height, AB::F::TWO);

        // incorporate row

        builder
            .when(incorporate_row)
            .when(AB::Expr::ONE - end_top_level)
            .assert_one(next.incorporate_sibling);

        let row_hash = std::array::from_fn(|i| {
            (start_top_level * left_output[i])
                + ((AB::Expr::ONE - start_top_level) * right_input[i])
        });

        self.internal_bus.interact(
            builder,
            true,
            incorporate_row,
            timestamp_after_initial_reads.clone(),
            end_timestamp - AB::F::TWO,
            opened_base_pointer,
            opened_element_size_inv,
            initial_opened_index,
            final_opened_index,
            row_hash,
        );

        for i in 0..CHUNK {
            builder
                .when(AB::Expr::ONE - end.clone())
                .when(next.incorporate_row)
                .assert_eq(next_left_input[i], left_output[i]);
        }

        builder
            .when(end_top_level)
            .when(incorporate_row)
            .assert_eq(final_opened_index, opened_length - AB::F::ONE);

        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, dim_base_pointer + initial_opened_index),
                [height],
                end_timestamp - AB::F::TWO,
                &read_initial_height_or_root_is_on_right,
            )
            .eval(builder, incorporate_row);
        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, dim_base_pointer + final_opened_index),
                [height],
                end_timestamp - AB::F::ONE,
                &read_final_height_or_sibling_array_start,
            )
            .eval(builder, incorporate_row);

        // incorporate sibling

        builder
            .when(incorporate_sibling)
            .when(AB::Expr::ONE - end_top_level)
            .assert_one(next.incorporate_row + next.incorporate_sibling);
        builder
            .when(end_top_level)
            .when(incorporate_sibling)
            .assert_eq(initial_opened_index, opened_length);

        builder
            .when(incorporate_sibling)
            .assert_eq(final_opened_index + AB::F::ONE, initial_opened_index);

        self.memory_bridge
            .read(
                MemoryAddress::new(self.address_space, index_base_pointer + proof_index),
                [root_is_on_right],
                timestamp_after_initial_reads.clone(),
                &read_initial_height_or_root_is_on_right,
            )
            .eval(builder, incorporate_sibling);
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    self.address_space,
                    sibling_base_pointer + (proof_index * AB::F::TWO),
                ),
                [sibling_array_start],
                timestamp_after_initial_reads.clone() + AB::F::ONE,
                &read_final_height_or_sibling_array_start,
            )
            .eval(builder, incorporate_sibling);

        for i in 0..CHUNK {
            builder
                .when(next.incorporate_sibling)
                .when(next_top_level_specific.root_is_on_right)
                .assert_eq(next_right_input[i], left_output[i]);
            builder
                .when(next.incorporate_sibling)
                .when(AB::Expr::ONE - next_top_level_specific.root_is_on_right)
                .assert_eq(next_left_input[i], left_output[i]);

            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        self.address_space,
                        sibling_array_start + AB::F::from_canonical_usize(i),
                    ),
                    [(root_is_on_right * left_input[i])
                        + ((AB::Expr::ONE - root_is_on_right) * right_input[i])],
                    timestamp_after_initial_reads.clone() + AB::F::from_canonical_usize(2 + i),
                    &reads[i],
                )
                .eval(builder, incorporate_sibling);
        }

        builder.when(incorporate_sibling).assert_eq(
            end_timestamp,
            timestamp_after_initial_reads + AB::F::from_canonical_usize(2 + CHUNK),
        );

        //// simple permute

        let simple_permute_specific: &SimplePoseidonSpecificCols<AB::Var> =
            specific[..SimplePoseidonSpecificCols::<AB::Var>::width()].borrow();

        let &SimplePoseidonSpecificCols {
            pc,
            is_compress,
            output_register,
            input_register_1,
            input_register_2,
            register_address_space,
            data_address_space,
            output_pointer,
            input_pointer_1,
            input_pointer_2,
            read_output_pointer,
            read_input_pointer_1,
            read_input_pointer_2,
            read_data_1,
            read_data_2,
            write_data_1,
            write_data_2,
        } = simple_permute_specific;

        let is_permute = AB::Expr::ONE - is_compress;

        self.execution_bridge
            .execute_and_increment_pc(
                (is_permute.clone() * AB::F::from_canonical_usize(PERM_POS2 as usize))
                    + (is_compress * AB::F::from_canonical_usize(COMP_POS2 as usize))
                    + AB::F::from_canonical_usize(self.simple_offset),
                [
                    output_register.into(),
                    input_register_1.into(),
                    input_register_2.into(),
                    register_address_space.into(),
                    data_address_space.into(),
                ],
                ExecutionState::new(pc, start_timestamp),
                AB::Expr::from_canonical_u32(NUM_SIMPLE_ACCESSES),
            )
            .eval(builder, simple);

        self.memory_bridge
            .read(
                MemoryAddress::new(register_address_space, output_register),
                [output_pointer],
                start_timestamp,
                &read_output_pointer,
            )
            .eval(builder, simple);

        self.memory_bridge
            .read(
                MemoryAddress::new(register_address_space, input_register_1),
                [input_pointer_1],
                start_timestamp + AB::F::ONE,
                &read_input_pointer_1,
            )
            .eval(builder, simple);

        self.memory_bridge
            .read(
                MemoryAddress::new(register_address_space, input_register_2),
                [input_pointer_2],
                start_timestamp + AB::F::TWO,
                &read_input_pointer_2,
            )
            .eval(builder, simple * is_compress);
        builder.when(simple).when(is_permute.clone()).assert_eq(
            input_pointer_2,
            input_pointer_1 + AB::F::from_canonical_usize(CHUNK),
        );

        self.memory_bridge
            .read(
                MemoryAddress::new(data_address_space, input_pointer_1),
                left_input,
                start_timestamp + AB::F::from_canonical_usize(3),
                &read_data_1,
            )
            .eval(builder, simple);

        self.memory_bridge
            .read(
                MemoryAddress::new(data_address_space, input_pointer_2),
                right_input,
                start_timestamp + AB::F::from_canonical_usize(4),
                &read_data_2,
            )
            .eval(builder, simple);

        self.memory_bridge
            .write(
                MemoryAddress::new(data_address_space, output_pointer),
                left_output,
                start_timestamp + AB::F::from_canonical_usize(5),
                &write_data_1,
            )
            .eval(builder, simple);

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    data_address_space,
                    output_pointer + AB::F::from_canonical_usize(CHUNK),
                ),
                right_output,
                start_timestamp + AB::F::from_canonical_usize(6),
                &write_data_2,
            )
            .eval(builder, simple * is_permute);
    }
}

impl VerifyBatchBus {
    #[allow(clippy::too_many_arguments)]
    pub fn interact<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        send: bool,
        multiplicity: impl Into<AB::Expr>,
        start_timestamp: impl Into<AB::Expr>,
        end_timestamp: impl Into<AB::Expr>,
        opened_base_pointer: impl Into<AB::Expr>,
        opened_element_size_inv: impl Into<AB::Expr>,
        initial_opened_index: impl Into<AB::Expr>,
        final_opened_index: impl Into<AB::Expr>,
        hash: [impl Into<AB::Expr>; CHUNK],
    ) {
        let mut fields = vec![
            start_timestamp.into(),
            end_timestamp.into(),
            opened_base_pointer.into(),
            opened_element_size_inv.into(),
            initial_opened_index.into(),
            final_opened_index.into(),
        ];
        fields.extend(hash.into_iter().map(Into::into));
        builder.push_interaction(
            self.0,
            fields,
            multiplicity.into(),
            if send {
                InteractionType::Send
            } else {
                InteractionType::Receive
            },
        );
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VerifyBatchBus(pub usize);