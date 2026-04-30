use std::{array::from_fn, borrow::Borrow};

use itertools::{izip, Itertools};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, DEFAULT_BLOCK_SIZE},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    utils::{assert_array_eq, not},
    ColumnsAir,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
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
use p3_field::PrimeField32;

use crate::{
    canonicity::{CanonicityAuxCols, CanonicitySubAir},
    count::DeferralCircuitCountBus,
    poseidon2::DeferralPoseidon2Bus,
    utils::{
        byte_commit_to_f, bytes_to_f, combine_output, split_memory_ops, COMMIT_NUM_BYTES,
        DIGEST_MEMORY_OPS, F_NUM_BYTES, OUTPUT_TOTAL_BYTES, OUTPUT_TOTAL_MEMORY_OPS,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
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
    pub output_len: [T; F_NUM_BYTES],
    pub output_commit_and_len_aux: [MemoryReadAuxCols<T>; OUTPUT_TOTAL_MEMORY_OPS],

    // Auxiliary columns to ensure the canonicity of each F byte decomposition in
    // output_commit.
    pub output_commit_lt_aux: [CanonicityAuxCols<T>; DIGEST_SIZE],

    // Initial [def_idx, output_len, 0, ...] digest on the first row; on non-first
    // rows bytes raw_output[local_idx * DIGEST_SIZE..(local_idx + 1) * DIGEST_SIZE]
    // written to memory and auxiliary columns.
    pub sponge_inputs: [T; DIGEST_SIZE],
    pub write_bytes_aux: [MemoryWriteAuxCols<T, DEFAULT_BLOCK_SIZE>; DIGEST_MEMORY_OPS],

    // Capacity of the permutation of write_bytes and the previous row's capacity on
    // non-last rows, compression on the last row.
    pub poseidon2_res: [T; DIGEST_SIZE],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralOutputAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub count_bus: DeferralCircuitCountBus,
    pub poseidon2_bus: DeferralPoseidon2Bus,
    pub bitwise_bus: BitwiseOperationLookupBus,
    pub address_bits: usize,
}

impl<F> BaseAir<F> for DeferralOutputAir {
    fn width(&self) -> usize {
        DeferralOutputCols::<F>::width()
    }
}
impl<F> BaseAirWithPublicValues<F> for DeferralOutputAir {}
impl<F> PartitionedBaseAir<F> for DeferralOutputAir {}
impl<F> ColumnsAir<F> for DeferralOutputAir {}

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

        // Constrain the canonicity of output_commit and output_len, i.e. that every
        // F_NUM_BYTES bytes uniquely represents an element of F.
        let output_commit_rcs = izip!(
            local.output_commit.chunks_exact(F_NUM_BYTES),
            local.output_commit_lt_aux
        )
        .map(|(bytes, aux)| {
            CanonicitySubAir.assert_canonicity(builder, bytes, &aux, local.is_first.into())
        })
        .collect_vec();

        for rc_pair in output_commit_rcs.chunks_exact(2) {
            self.bitwise_bus
                .send_range(rc_pair[0].clone(), rc_pair[1].clone())
                .eval(builder, local.is_first);
        }

        // Constrain the consistency of current_commit_state at each point in this
        // section's rows. The initial state should be [deferral_idx, output_len,
        // ..., 0].
        let output_len = bytes_to_f(&local.output_len);
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
        // section_idx * DIGEST_SIZE.
        let mut when_last = builder.when(local.is_last);

        when_last.assert_eq(
            output_len,
            local.section_idx * AB::Expr::from_usize(DIGEST_SIZE),
        );
        assert_array_eq(
            &mut when_last,
            byte_commit_to_f(&local.output_commit),
            local.poseidon2_res,
        );

        // Constrain poseidon2_res is the running permute capacity on all non-last rows,
        // and the compression on the last row.
        let rhs = from_fn(|i| is_transition.clone() * local.poseidon2_res[i]);
        self.poseidon2_bus
            .lookup(next.sponge_inputs, rhs, next.poseidon2_res, next.is_last)
            .eval(builder, next.is_valid);

        // We range check the top byte of both heap pointers to ensure that each access
        // is in [0, 2^address_bits). The memory merkle argument ensures each pointer
        // is less than 2^addr_bits, and this range check ensures the decomposition is
        // canonical. Note that constraining the starting output pointer is sufficient
        // to constrain the entire write is in range - even if output_ptr + output_len
        // wraps, there will be several written values in the middle that do not.
        debug_assert!(RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS >= self.address_bits);
        let limb_shift =
            AB::F::from_usize(1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.address_bits));

        self.bitwise_bus
            .send_range(
                local.rd_val[RV32_REGISTER_NUM_LIMBS - 1] * limb_shift,
                local.rs_val[RV32_REGISTER_NUM_LIMBS - 1] * limb_shift,
            )
            .eval(builder, local.is_first);

        // We also constrain output_len to be under 2^address_bits.
        self.bitwise_bus
            .send_range(
                local.output_len[RV32_REGISTER_NUM_LIMBS - 1] * limb_shift,
                AB::Expr::ZERO,
            )
            .eval(builder, local.is_first);

        // Constrain the heap pointer memory reads.
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
        let output_len_full = from_fn(|i| {
            if i < F_NUM_BYTES {
                local.output_len[i].into()
            } else {
                AB::Expr::ZERO
            }
        });
        let output_commit_and_len =
            combine_output(local.output_commit.map(Into::into), output_len_full);
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
                        input_ptr.clone() + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    local.from_state.timestamp + AB::Expr::from_usize(2 + chunk_idx),
                    aux,
                )
                .eval(builder, local.is_first);
        }

        let write_bytes_chunks =
            split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(local.sponge_inputs);
        let section_idx_minus_one = local.section_idx - AB::Expr::ONE;

        for (chunk_idx, (data, aux)) in write_bytes_chunks
            .into_iter()
            .zip(&local.write_bytes_aux)
            .enumerate()
        {
            for bytes in data.chunks(2) {
                self.bitwise_bus
                    .send_range(bytes[0], bytes[1])
                    .eval(builder, local.is_valid - local.is_first);
            }

            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        e.clone(),
                        output_ptr.clone()
                            + (section_idx_minus_one.clone() * AB::Expr::from_usize(DIGEST_SIZE))
                            + AB::Expr::from_usize(chunk_idx * DEFAULT_BLOCK_SIZE),
                    ),
                    data,
                    local.from_state.timestamp
                        + AB::Expr::from_usize(2 + OUTPUT_TOTAL_MEMORY_OPS + chunk_idx)
                        + (section_idx_minus_one.clone() * AB::Expr::from_usize(DIGEST_MEMORY_OPS)),
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
                (local.section_idx * AB::Expr::from_usize(DIGEST_MEMORY_OPS))
                    + AB::Expr::from_usize(OUTPUT_TOTAL_MEMORY_OPS + 2),
                (DEFAULT_PC_STEP, None),
            )
            .eval(builder, local.is_last);
    }
}
