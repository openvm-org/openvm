use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::utils::{and, assert_array_eq, not, or};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
    LocalOpcode,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{
    count::bus::DeferralCircuitCountBus,
    poseidon2::bus::DeferralPoseidon2Bus,
    utils::{
        byte_commit_to_f, bytes_to_f, combine_output, split_memory_ops, COMMIT_NUM_BYTES,
        DIGEST_MEMORY_OPS, MEMORY_OP_SIZE, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralOutputCols<T> {
    // Indicates the status of this row, i.e. if it is valid and where it is in a
    // section of rows that correspond to a single opcode invocation
    pub is_valid: T,
    pub is_first: T,
    pub section_idx: T,

    // Initial execution state + instruction operands
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs_ptr: T,
    pub deferral_idx: T,

    // Heap pointers + auxiliary read columns
    pub rd_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rs_val: [T; RV32_REGISTER_NUM_LIMBS],
    pub rd_aux: MemoryReadAuxCols<T>,
    pub rs_aux: MemoryReadAuxCols<T>,

    // Read data and auxiliary columns. output_commit and output_len are read
    // contiguously from heap with layout [output_commit || output_len].
    // The onion hash of all bytes written by this opcode invocation is
    // constrained to output_commit.
    pub output_commit: [T; COMMIT_NUM_BYTES],
    pub output_len: [T; RV32_REGISTER_NUM_LIMBS],
    pub output_commit_and_len_aux: [MemoryReadAuxCols<T>; OUTPUT_TOTAL_MEMORY_OPS],

    // Bytes raw_output[local_idx * DIGEST_SIZE..(local_idx + 1) * DIGEST_SIZE]
    // written to memory and auxiliary columns
    pub write_bytes: [T; DIGEST_SIZE],
    pub write_bytes_aux: [MemoryWriteAuxCols<T, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],

    // Running hash of this section's write_bytes, constrained to be output_commit;
    // note the initial state should be [deferral_idx, 0, ..., 0]
    pub current_commit_state: [T; DIGEST_SIZE],
}

// TODO: This should probably be split into two AIRs, one for the read + execution
// state and one for the reads. This is quite difficult to do currently, though.
#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
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
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let next = main.row_slice(1).expect("window should have two elements");
        let local: &DeferralOutputCols<AB::Var> = (*local).borrow();
        let next: &DeferralOutputCols<AB::Var> = (*next).borrow();

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

        builder
            .when(not(local.is_valid))
            .assert_zero(local.is_first);
        builder
            .when(not(local.is_valid))
            .assert_zero(local.section_idx);

        builder.when(local.is_first).assert_zero(local.section_idx);
        builder
            .when(next.section_idx)
            .assert_eq(next.section_idx, local.section_idx + AB::Expr::ONE);
        builder
            .when(and(next.is_valid, not(next.is_first)))
            .assert_eq(next.section_idx, local.section_idx + AB::Expr::ONE);

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

        // Constrain the consistency of current_commit_state at each point in this
        // section's rows. The initial state should be [deferral_idx, 0, ..., 0]
        // and the final state should be output_commit. Note that output_len must
        // be divisible by DIGEST_SIZE.
        let is_last_or_invalid = or(next.is_first, not(next.is_valid));
        let mut when_last_or_invalid = builder.when(is_last_or_invalid.clone());

        when_last_or_invalid.assert_eq(
            bytes_to_f(&local.output_len),
            (local.section_idx + local.is_valid) * AB::Expr::from_usize(DIGEST_SIZE),
        );
        assert_array_eq(
            &mut when_last_or_invalid,
            byte_commit_to_f(&local.output_commit),
            local.current_commit_state,
        );

        self.count_bus
            .send(local.deferral_idx)
            .eval(builder, local.is_first);

        let mut initial_state = [AB::Expr::ZERO; DIGEST_SIZE];
        initial_state[0] = local.deferral_idx.into();

        self.poseidon2_bus
            .compress(initial_state, local.write_bytes, local.current_commit_state)
            .eval(builder, local.is_first);

        self.poseidon2_bus
            .compress(
                local.current_commit_state,
                next.write_bytes,
                next.current_commit_state,
            )
            .eval(
                builder,
                local.is_valid * and(next.is_valid, not(next.is_first)),
            );

        // Constrain the heap pointer memory reads first.
        let d = AB::Expr::from_u32(RV32_REGISTER_AS);
        let e = AB::Expr::from_u32(RV32_MEMORY_AS);

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), local.rd_ptr),
                local.rd_val,
                local.from_state.timestamp,
                &local.rd_aux,
            )
            .eval(builder, local.is_first);

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), local.rs_ptr),
                local.rs_val,
                local.from_state.timestamp + AB::Expr::ONE,
                &local.rs_aux,
            )
            .eval(builder, local.is_first);

        // Constrain memory reads and writes using the MemoryBridge. a and b are
        // register pointers whose values are read first, then used as heap
        // pointers. c carries deferral_idx.
        let input_ptr = bytes_to_f(&local.rs_val);
        let output_ptr = bytes_to_f(&local.rd_val);
        let output_commit_and_len = combine_output(local.output_commit, local.output_len);
        let output_commit_and_len_chunks =
            split_memory_ops::<_, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS>(
                output_commit_and_len,
            );

        for (chunk_idx, (data, aux)) in output_commit_and_len_chunks
            .into_iter()
            .zip(&local.output_commit_and_len_aux)
            .enumerate()
        {
            self.memory_bridge
                .read(
                    MemoryAddress::new(
                        e.clone(),
                        input_ptr.clone() + AB::Expr::from_usize(chunk_idx * MEMORY_OP_SIZE),
                    ),
                    data,
                    local.from_state.timestamp + AB::Expr::from_usize(2 + chunk_idx),
                    aux,
                )
                .eval(builder, local.is_first);
        }

        let write_bytes_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(local.write_bytes);

        for (chunk_idx, (data, aux)) in write_bytes_chunks
            .into_iter()
            .zip(&local.write_bytes_aux)
            .enumerate()
        {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e.clone(),
                        output_ptr.clone()
                            + (local.section_idx.into() * AB::Expr::from_usize(DIGEST_SIZE))
                            + AB::Expr::from_usize(chunk_idx * MEMORY_OP_SIZE),
                    ),
                    data,
                    local.from_state.timestamp
                        + AB::Expr::from_usize(2 + OUTPUT_TOTAL_MEMORY_OPS + chunk_idx)
                        + (local.section_idx.into() * AB::Expr::from_usize(DIGEST_MEMORY_OPS)),
                    aux,
                )
                .eval(builder, local.is_valid);
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
                (local.section_idx.into() * AB::Expr::from_usize(DIGEST_MEMORY_OPS))
                    + AB::Expr::from_usize(OUTPUT_TOTAL_MEMORY_OPS + DIGEST_MEMORY_OPS + 2),
                (DEFAULT_PC_STEP, None),
            )
            .eval(builder, local.is_valid * is_last_or_invalid);
    }
}
