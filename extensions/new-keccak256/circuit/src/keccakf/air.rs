use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{bitwise_op_lookup::BitwiseOperationLookupBus, utils::not};
use openvm_instructions::riscv::{
    RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS,
};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder,
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use p3_keccak_air::{KeccakAir, NUM_ROUNDS, U64_LIMBS};

use crate::keccakf::columns::{KeccakfVmCols, NUM_KECCAKF_VM_COLS, NUM_KECCAK_PERM_COLS};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for KeccakfVmAir {}
impl<F> PartitionedBaseAir<F> for KeccakfVmAir {}
impl<F> BaseAir<F> for KeccakfVmAir {
    fn width(&self) -> usize {
        NUM_KECCAKF_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for KeccakfVmAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let (local, next) = (main.row_slice(0), main.row_slice(1));
        let local: &KeccakfVmCols<_> = (*local).borrow();
        let next: &KeccakfVmCols<_> = (*next).borrow();

        let mut timestamp: AB::Expr = local.timestamp.into();
        let mem_oc = &local.mem_oc;

        // only active during the first round in each instruction
        self.eval_instruction(builder, local, &mut timestamp, &mem_oc.register_aux_cols);
        // only active during the first round in each instruction
        self.constrain_input_read(
            builder,
            local,
            &mut timestamp,
            &mem_oc.buffer_bytes_read_aux_cols,
        );

        let keccak_f_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_PERM_COLS);
        keccak_f_air.eval(&mut sub_builder);

        // only active during the last round in each instruction
        self.constrain_output_write(
            builder,
            local,
            &mut timestamp,
            &mem_oc.buffer_bytes_write_aux_cols,
        );

        self.constrain_rounds_transition(builder, local, next, timestamp);
    }
}

impl KeccakfVmAir {
    #[inline]
    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_timestamp: &mut AB::Expr,
        register_aux: &[MemoryReadAuxCols<AB::Var>; 1],
    ) {
        let instruction = local.instruction;
        let is_enabled = instruction.is_enabled;
        builder.assert_bool(is_enabled);
        let is_first_round = local.inner.step_flags[0];
        let is_final_round = local.inner.step_flags[NUM_ROUNDS - 1];
        let is_enabled_is_first_round = is_enabled * is_first_round;
        let is_enabled_is_final_round = is_enabled * is_final_round;
        builder.assert_eq(is_enabled_is_first_round, is_first_round * is_enabled);
        builder.assert_eq(is_enabled_is_final_round, is_final_round * is_enabled);

        let rd_ptr = instruction.rd_ptr;
        let buffer_ptr_limbs = instruction.buffer_ptr_limbs.map(Into::into);
        // 50 buffer reads and 50 buffer writes and 1 register read
        let timestamp_change =
            local.is_enabled_is_first_round * AB::F::from_canonical_u32(1 + 50 + 50);

        // safety: it is safe to use timestamp instead of storing a separate
        // instruction.start_timestamp because the execution_bridge interaction is only
        // enabled in the first row, so timestamp = start_timestamp since there is no more
        // timestamp increments
        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(KeccakfOpcode::KECCAKF as usize + self.offset),
                [
                    rd_ptr.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, local.timestamp),
                timestamp_change,
            )
            .eval(builder, local.is_enabled_is_first_round);

        self.memory_bridge
            .read(
                MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_REGISTER_AS), rd_ptr),
                buffer_ptr_limbs,
                start_timestamp.clone(),
                &register_aux[0],
            )
            .eval(builder, local.is_enabled_is_first_round);
        *start_timestamp += local.is_enabled_is_first_round.into();

        builder.assert_eq(
            instruction.buffer_ptr,
            instruction.buffer_ptr_limbs[0]
                + instruction.buffer_ptr_limbs[1] * AB::F::from_canonical_u32(1 << 8)
                + instruction.buffer_ptr_limbs[2] * AB::F::from_canonical_u32(1 << 16)
                + instruction.buffer_ptr_limbs[3] * AB::F::from_canonical_u32(1 << 24),
        );

        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        let need_range_check = [
            *instruction.buffer_ptr_limbs.last().unwrap(),
            *instruction.buffer_ptr_limbs.last().unwrap(),
        ];
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, local.is_enabled_is_first_round);
        }
    }

    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_timestamp: &mut AB::Expr,
        buffer_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 200 / 4],
    ) {
        const PREIMAGE_BYTES: usize = 25 * U64_LIMBS * 2;
        let local_preimage_bytes: [AB::Expr; PREIMAGE_BYTES] = std::array::from_fn(|byte_idx| {
            // `preimage` is represented as 5 * 5 * U64_LIMBS u16 limbs; each u16 limb is split into
            // 2 bytes.
            let u16_idx = byte_idx / 2;
            let is_hi_byte = (byte_idx % 2) == 1;

            let i = u16_idx / U64_LIMBS;
            let limb = u16_idx % U64_LIMBS;

            let y = i / 5;
            let x = i % 5;

            let state_limb: AB::Expr = local.inner.preimage[y][x][limb].into();
            let hi: AB::Expr = local.preimage_state_hi[i * U64_LIMBS + limb].into();
            let lo: AB::Expr = state_limb - hi.clone() * AB::F::from_canonical_u64(1 << 8);

            if is_hi_byte {
                hi
            } else {
                lo
            }
        });

        for idx in 0..(PREIMAGE_BYTES / 4) {
            let read_chunk: [AB::Expr; 4] =
                std::array::from_fn(|j| local_preimage_bytes[4 * idx + j].clone());

            let ptr = local.instruction.buffer_ptr + AB::Expr::from_canonical_usize(idx * 4);

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    read_chunk,
                    start_timestamp.clone(),
                    &buffer_bytes_read_aux_cols[idx],
                )
                .eval(builder, local.is_enabled_is_first_round);

            *start_timestamp += local.is_enabled_is_first_round.into();
        }
    }

    #[inline]
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_timestamp: &mut AB::Expr,
        buffer_bytes_write_aux_cols: &[MemoryWriteAuxCols<AB::Var, 4>; 200 / 4],
    ) {
        const POSTIMAGE_BYTES: usize = 25 * 4 * 2;
        let local_postimage_bytes: [AB::Expr; POSTIMAGE_BYTES] = std::array::from_fn(|byte_idx| {
            // `preimage` is represented as 5 * 5 * U64_LIMBS u16 limbs; each u16 limb is split into
            // 2 bytes.
            let u16_idx = byte_idx / 2;
            let is_hi_byte = (byte_idx % 2) == 1;

            let i = u16_idx / U64_LIMBS;
            let limb = u16_idx % U64_LIMBS;
            let y = i / 5;
            let x = i % 5;

            let state_limb: AB::Expr = local.inner.a_prime_prime_prime(y, x, limb).into();
            let hi: AB::Expr = local.postimage_state_hi[i * U64_LIMBS + limb].into();
            let lo: AB::Expr = state_limb - hi.clone() * AB::F::from_canonical_u64(1 << 8);

            if is_hi_byte {
                hi
            } else {
                lo
            }
        });

        for idx in 0..(200 / 4) {
            let write_chunk: [AB::Expr; 4] =
                std::array::from_fn(|j| local_postimage_bytes[4 * idx + j].clone());

            let ptr = local.instruction.buffer_ptr + AB::Expr::from_canonical_usize(idx * 4);

            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    write_chunk,
                    start_timestamp.clone(),
                    &buffer_bytes_write_aux_cols[idx],
                )
                .eval(builder, local.is_enabled_is_final_round);

            *start_timestamp += local.is_enabled_is_final_round.into();
        }
    }

    // responsible for constraining everything that needs to be constrained between rows in the same
    // instruction but different keccakf rounds
    #[inline]
    pub fn constrain_rounds_transition<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        next: &KeccakfVmCols<AB::Var>,
        timestamp: AB::Expr,
    ) {
        let is_final_round = local.inner.step_flags[NUM_ROUNDS - 1];
        let is_enabled = local.instruction.is_enabled;
        let need_check = is_enabled * not(is_final_round);

        for idx in 0..100 {
            builder
                .when(need_check.clone())
                .assert_eq(local.preimage_state_hi[idx], next.preimage_state_hi[idx]);
            builder
                .when(need_check.clone())
                .assert_eq(local.postimage_state_hi[idx], next.postimage_state_hi[idx]);
        }
        builder
            .when(need_check.clone())
            .assert_eq(local.instruction.pc, next.instruction.pc);
        builder
            .when(need_check.clone())
            .assert_eq(local.instruction.is_enabled, next.instruction.is_enabled);
        builder
            .when(need_check.clone())
            .assert_eq(local.instruction.rd_ptr, next.instruction.rd_ptr);
        builder
            .when(need_check.clone())
            .assert_eq(local.instruction.buffer_ptr, next.instruction.buffer_ptr);
        for limb in 0..4 {
            builder.when(need_check.clone()).assert_eq(
                local.instruction.buffer_ptr_limbs[limb],
                next.instruction.buffer_ptr_limbs[limb],
            );
        }
        // safety: mem_oc does not need to be checked here because in the rows which is not first or
        // last it is not used and in the first or last rows, the mem_oc fields which is
        // used is already constraiend by the interactions
        builder
            .when(need_check.clone())
            .assert_eq(timestamp, next.timestamp);
    }
}
