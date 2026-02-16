use std::borrow::Borrow;

use itertools::fold;
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
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV32_CELL_BITS, LocalOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::utils::{byte_commit_to_f, COMMIT_NUM_BYTES};

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
    pub a: T,
    pub b: T,
    pub c: T,
    pub deferral_idx: T,

    // Read data and auxiliary columns, the onion hash of all bytes written by this
    // opcode invocation will be constrained to output_commit
    pub output_commit: [T; COMMIT_NUM_BYTES],
    pub output_len: [T; RV32_CELL_BITS],
    pub output_commit_aux: MemoryReadAuxCols<T>,
    pub output_len_aux: MemoryReadAuxCols<T>,

    // Bytes raw_output[local_idx * DIGEST_SIZE..(local_idx + 1) * DIGEST_SIZE]
    // written to memory and auxiliary columns
    pub write_bytes: [T; DIGEST_SIZE],
    pub write_bytes_aux: MemoryWriteAuxCols<T, DIGEST_SIZE>,

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
            .when(and(local.is_valid, next.is_valid))
            .assert_eq(next.section_idx, local.section_idx + AB::Expr::ONE);

        // Constrain that the read columns and other operands stay the same within a
        // section, i.e. when section_idx is non-zero. Note that the read auxiliary
        // columns are only used on the first row - thus we leave their consistency
        // unconstrained.
        let mut when_section_transition = builder.when(next.section_idx);

        when_section_transition.assert_eq(local.from_state.pc, next.from_state.pc);
        when_section_transition.assert_eq(local.from_state.timestamp, next.from_state.timestamp);
        when_section_transition.assert_eq(local.a, next.a);
        when_section_transition.assert_eq(local.b, next.b);
        when_section_transition.assert_eq(local.c, next.c);
        when_section_transition.assert_eq(local.deferral_idx, next.deferral_idx);

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
        let mut when_last_or_invalid = builder.when(or(next.is_first, not(next.is_valid)));

        when_last_or_invalid.assert_eq(
            fold(
                local.output_len.iter().enumerate(),
                AB::Expr::ZERO,
                |acc, (i, &b)| acc + (b * AB::Expr::from_usize(1 << i)),
            ),
            local.section_idx,
        );

        assert_array_eq(
            &mut when_last_or_invalid,
            byte_commit_to_f(&local.output_commit),
            local.current_commit_state,
        );

        // TODO: constrain that on the first row local.current_commit_state is the
        // poseidon2 compress of local.write_bytes and [deferral_idx, 0, ..., 0]

        // TODO: constrain that next.current_commit_state is the poseidon2 compress
        // of next.write_bytes and local.current_commit_state

        // Constrain memory reads and writes using the MemoryBridge. There are two
        // reads per opcode that are processed on the first row, and one write per
        // valid row. Address spaces d and e must be the RV32 memory and register
        // address spaces respectively.
        let d = AB::Expr::TWO;
        let e = AB::Expr::ONE;

        self.memory_bridge
            .read(
                MemoryAddress::new(d.clone(), local.b),
                local.output_commit,
                local.from_state.timestamp,
                &local.output_commit_aux,
            )
            .eval(builder, local.is_first);

        self.memory_bridge
            .read(
                MemoryAddress::new(e.clone(), local.c),
                local.output_len,
                local.from_state.timestamp + AB::Expr::ONE,
                &local.output_len_aux,
            )
            .eval(builder, local.is_first);

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    d.clone(),
                    local.a + (local.section_idx * AB::Expr::from_usize(DIGEST_SIZE)),
                ),
                local.write_bytes,
                local.from_state.timestamp + AB::Expr::TWO + local.section_idx,
                &local.write_bytes_aux,
            )
            .eval(builder, local.is_valid);

        // Evaluate the execution interaction. Because a single opcode spans many
        // rows, we only execute this on the last one.
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                AB::Expr::from_usize(DeferralOpcode::OUTPUT.global_opcode_usize()),
                [
                    local.a.into(),
                    local.b.into(),
                    local.c.into(),
                    d,
                    e,
                    local.deferral_idx.into(),
                ],
                local.from_state,
                local.section_idx + AB::Expr::from_u8(3),
                (DEFAULT_PC_STEP, None),
            )
            .eval(
                builder,
                local.is_valid * or(next.is_first, not(next.is_valid)),
            );
    }
}
