use std::{array::from_fn, borrow::Borrow};

use itertools::{izip, Itertools};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    utils::{assert_array_eq, not},
    var_range::VariableRangeCheckerBus,
    ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS},
    LocalOpcode,
};
use openvm_riscv_circuit::adapters::{byte_ptr_to_u16_ptr, expand_to_rv64_block};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use p3_field::PrimeField32;

use crate::{
    canonicity::{CanonicityAuxCols, CanonicitySubAir},
    count::DeferralCircuitCountBus,
    poseidon2::DeferralPoseidon2Bus,
    utils::{
        combine_output_cells, scale_output_len, scale_rv64_ptr_high_u16, split_cell_memory_ops,
        u16_commit_to_f, u16s_to_f, COMMIT_NUM_U16S, F_NUM_U16S, OUTPUT_TOTAL_MEMORY_OPS,
        OUTPUT_TOTAL_NUM_U16S, RV64_PTR_U16S, SPONGE_BYTES_PER_ROW, SPONGE_ROW_MEMORY_OPS,
        U16_BITS,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DeferralOutputCols<T> {
    // Indicates the status of this row, i.e. if it is valid and where it is in a
    // section of rows that correspond to a single opcode invocation
    pub is_valid: T,
    pub is_first: T,
    pub is_last: T,
    pub section_idx: T,

    // Initial execution state + instruction operands
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs_ptr: T,
    pub deferral_idx: T,

    // Heap pointers + auxiliary read columns.
    // Low 32 bits of heap pointers, packed as u16 cells.
    pub rd_val: [T; RV64_PTR_U16S],
    pub rs_val: [T; RV64_PTR_U16S],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // First row reads [output_commit || output_len_le] from heap as u16 cells.
    // `output_commit` is the onion hash of the output bytes.
    pub output_commit: [T; COMMIT_NUM_U16S],
    pub output_len: [T; F_NUM_U16S],
    pub output_commit_and_len_aux: [MemoryReadAuxCols<T>; OUTPUT_TOTAL_MEMORY_OPS],

    // Auxiliary columns to ensure canonicity of output_commit cells.
    pub output_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],

    // First row sponge input is [deferral_idx, output_len, 0, ...].
    // Later rows sponge and write the next SPONGE_BYTES_PER_ROW output bytes.
    pub sponge_inputs: [T; DIGEST_SIZE],
    pub write_bytes_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; SPONGE_ROW_MEMORY_OPS],

    // Running Poseidon2 capacity on non-last rows; final compression on the last row.
    pub poseidon2_res: [T; DIGEST_SIZE],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(DeferralOutputCols<u8>)]
pub struct DeferralOutputAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
    pub range_bus: VariableRangeCheckerBus,
    pub address_bits: usize,
}

impl<F> BaseAir<F> for DeferralOutputAir {
    fn width(&self) -> usize {
        DeferralOutputCols::<F>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralOutputAir {}
impl<F> PartitionedBaseAir<F> for DeferralOutputAir {}

impl<AB> Air<AB> for DeferralOutputAir
where
    AB: InteractionBuilder,
    AB::F: PrimeField32,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let next = main.row_slice(1).expect("window should have two elements");
        let local: &DeferralOutputCols<AB::Var> = (*local).borrow();
        let next: &DeferralOutputCols<AB::Var> = (*next).borrow();

        let is_transition = next.is_valid - next.is_first;
        let is_last = local.is_valid - is_transition.clone();

        // Constrain the status flags. Particularly, section_idx must (a) always
        // reset to 0 upon reaching a new section, and (b) otherwise increment by
        // one each valid row. Additionally, for convenience we constrain that
        // all valid rows must be at the top of the trace.
        builder.assert_bool(local.is_valid);
        builder.assert_bool(local.is_first);

        builder
            .when_transition()
            .assert_bool(local.is_valid - next.is_valid);
        builder
            .when_first_row()
            .when(local.is_valid)
            .assert_one(local.is_first);

        builder.assert_eq(local.is_last, is_last);

        builder
            .when(not(local.is_valid))
            .assert_zero(local.is_first);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.section_idx);

        builder.when(local.is_first).assert_zero(local.section_idx);
        builder
            .when(is_transition.clone())
            .assert_one(next.section_idx - local.section_idx);

        // Constrain that the read columns and other operands stay the same within a
        // section, i.e. when section_idx is non-zero. Note that the read auxiliary
        // columns are only used on the first row - thus we leave their consistency
        // unconstrained.
        let mut when_section_transition = builder.when(next.section_idx);

        when_section_transition.assert_eq(local.from_state.pc, next.from_state.pc);
        when_section_transition.assert_eq(local.from_state.timestamp, next.from_state.timestamp);
        when_section_transition.assert_eq(local.rd_ptr, next.rd_ptr);
        when_section_transition.assert_eq(local.rs_ptr, next.rs_ptr);
        when_section_transition.assert_eq(local.deferral_idx, next.deferral_idx);

        assert_array_eq(&mut when_section_transition, local.rd_val, next.rd_val);
        assert_array_eq(&mut when_section_transition, local.rs_val, next.rs_val);

        assert_array_eq(
            &mut when_section_transition,
            local.output_commit,
            next.output_commit,
        );
        assert_array_eq(
            &mut when_section_transition,
            local.output_len,
            next.output_len,
        );

        // Constrain the canonicity of output_commit, i.e. that every
        // F_NUM_U16S u16 cells uniquely represent an element of F.
        let output_commit_rcs = izip!(
            local.output_commit.chunks_exact(F_NUM_U16S),
            local.output_commit_lt_aux
        )
        .map(|(cells, aux)| {
            let cells: [_; F_NUM_U16S] = cells.try_into().unwrap();
            CanonicitySubAir.assert_canonicity(
                builder,
                cells.map(Into::into),
                &aux,
                local.is_first.into(),
            )
        })
        .collect_vec();

        for rc in output_commit_rcs {
            self.range_bus
                .range_check(rc, U16_BITS)
                .eval(builder, local.is_first);
        }

        // Range-check every u16 cell of `output_commit` and `output_len`.
        for &cell in local.output_commit.iter().chain(local.output_len.iter()) {
            self.range_bus
                .range_check(cell, U16_BITS)
                .eval(builder, local.is_first);
        }

        // Init-row sponge state: [deferral_idx, output_len_as_F, 0, ...].
        let output_len = u16s_to_f(&local.output_len);
        let mut initial_state = [AB::Expr::ZERO; DIGEST_SIZE];
        initial_state[0] = local.deferral_idx.into();
        initial_state[1] = output_len.clone();

        assert_array_eq(
            &mut builder.when(local.is_first),
            initial_state,
            local.sponge_inputs,
        );

        self.count_bus
            .send(local.deferral_idx)
            .eval(builder, local.is_first);

        // The final state should be output_commit, and output_len must be the final
        // section_idx * SPONGE_BYTES_PER_ROW.
        let mut when_last = builder.when(local.is_last);

        when_last.assert_eq(
            output_len,
            local.section_idx * AB::Expr::from_usize(SPONGE_BYTES_PER_ROW),
        );
        assert_array_eq(
            &mut when_last,
            u16_commit_to_f(&local.output_commit),
            local.poseidon2_res,
        );

        // Constrain poseidon2_res is the running permute capacity on all non-last rows,
        // and the compression on the last row.
        let rhs = from_fn(|i| is_transition.clone() * local.poseidon2_res[i]);
        self.poseidon2_bus
            .lookup(next.sponge_inputs, rhs, next.poseidon2_res, next.is_last)
            .eval(builder, next.is_valid);

        // We range check the high u16 of both heap pointers to ensure that each
        // access is in [0, 2^address_bits).
        self.range_bus
            .range_check(
                scale_rv64_ptr_high_u16::<AB::Expr, _>(
                    local.rd_val[RV64_PTR_U16S - 1],
                    self.address_bits,
                ),
                U16_BITS,
            )
            .eval(builder, local.is_first);
        self.range_bus
            .range_check(
                scale_rv64_ptr_high_u16::<AB::Expr, _>(
                    local.rs_val[RV64_PTR_U16S - 1],
                    self.address_bits,
                ),
                U16_BITS,
            )
            .eval(builder, local.is_first);

        // Constrain output_len to be under 2^address_bits also.
        self.range_bus
            .range_check(
                scale_output_len::<AB::Expr, _>(&local.output_len, self.address_bits),
                U16_BITS,
            )
            .eval(builder, local.is_first);

        // Constrain the heap pointer memory reads.
        let d = AB::Expr::from_u32(RV64_REGISTER_AS);
        let e = AB::Expr::from_u32(RV64_MEMORY_AS);

        let rd_bus: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&local.rd_val);
        let rs_bus: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&local.rs_val);

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), byte_ptr_to_u16_ptr::<AB>(local.rd_ptr)),
                rd_bus,
                local.from_state.timestamp,
                &local.rd_aux,
            )
            .eval(builder, local.is_first);

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), byte_ptr_to_u16_ptr::<AB>(local.rs_ptr)),
                rs_bus,
                local.from_state.timestamp + AB::Expr::ONE,
                &local.rs_aux,
            )
            .eval(builder, local.is_first);

        // Constrain memory reads and writes using the MemoryBridge. a and b are
        // register pointers whose values are read first, then used as heap
        // pointers. c carries deferral_idx.
        let input_ptr = u16s_to_f(&local.rs_val);
        let output_ptr = u16s_to_f(&local.rd_val);

        let output_len_full: [AB::Expr; BLOCK_FE_WIDTH] = expand_to_rv64_block(&local.output_len);

        let output_commit_and_len =
            combine_output_cells(local.output_commit.map(Into::into), output_len_full);
        let output_chunks =
            split_cell_memory_ops::<AB::Expr, OUTPUT_TOTAL_NUM_U16S, OUTPUT_TOTAL_MEMORY_OPS>(
                output_commit_and_len,
            );

        for (chunk_idx, (data, aux)) in output_chunks
            .into_iter()
            .zip(&local.output_commit_and_len_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e.clone(),
                        byte_ptr_to_u16_ptr::<AB>(
                            input_ptr.clone()
                                + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    data,
                    local.from_state.timestamp + AB::Expr::from_usize(2 + chunk_idx),
                    aux,
                )
                .eval(builder, local.is_first);
        }

        // Data-row sponge cells must be canonical u16s; init-row cells are
        // pinned by `assert_array_eq` above against the already-canonical
        // `deferral_idx` / `output_len` columns.
        for &cell in &local.sponge_inputs {
            self.range_bus
                .range_check(cell, U16_BITS)
                .eval(builder, local.is_valid - local.is_first);
        }

        // Each data row writes `sponge_inputs` to memory as
        // `SPONGE_ROW_MEMORY_OPS` `BLOCK_FE_WIDTH`-cell bus blocks.
        let section_idx_minus_one = local.section_idx - AB::Expr::ONE;
        let write_cell_chunks = split_cell_memory_ops::<AB::Expr, DIGEST_SIZE, SPONGE_ROW_MEMORY_OPS>(
            local.sponge_inputs.map(Into::into),
        );
        for (chunk_idx, (data, aux)) in write_cell_chunks
            .into_iter()
            .zip(&local.write_bytes_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e.clone(),
                        byte_ptr_to_u16_ptr::<AB>(
                            output_ptr.clone()
                                + (section_idx_minus_one.clone()
                                    * AB::Expr::from_usize(SPONGE_BYTES_PER_ROW))
                                + AB::Expr::from_usize(chunk_idx * MEMORY_BLOCK_BYTES),
                        ),
                    ),
                    data,
                    local.from_state.timestamp
                        + AB::Expr::from_usize(2 + OUTPUT_TOTAL_MEMORY_OPS + chunk_idx)
                        + (section_idx_minus_one.clone()
                            * AB::Expr::from_usize(SPONGE_ROW_MEMORY_OPS)),
                    aux,
                )
                .eval(builder, local.is_valid - local.is_first);
        }

        // Evaluate the execution interaction. Because a single opcode spans many
        // rows, we only execute this on the last one.
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                AB::Expr::from_usize(DeferralOpcode::OUTPUT.global_opcode_usize()),
                [
                    local.rd_ptr.into(),
                    local.rs_ptr.into(),
                    local.deferral_idx.into(),
                    d,
                    e,
                ],
                local.from_state,
                (local.section_idx * AB::Expr::from_usize(SPONGE_ROW_MEMORY_OPS))
                    + AB::Expr::from_usize(OUTPUT_TOTAL_MEMORY_OPS + 2),
                (DEFAULT_PC_STEP, None),
            )
            .eval(builder, local.is_last);
    }
}
