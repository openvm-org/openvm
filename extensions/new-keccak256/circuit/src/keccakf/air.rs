use std::{borrow::Borrow, iter::once};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
};
use openvm_instructions::riscv::RV32_MEMORY_AS;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use strum::IntoEnumIterator;
use crate::keccakf::columns::{KeccakfVmCols, NUM_KECCAKF_VM_COLS, NUM_KECCAK_PERM_COLS};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_instructions::riscv::RV32_REGISTER_AS;
use openvm_stark_backend::interaction::PermutationCheckBus;
use p3_keccak_air::U64_LIMBS;
use p3_keccak_air::KeccakAir;
use openvm_instructions::riscv::RV32_CELL_BITS;
use openvm_instructions::riscv::RV32_REGISTER_NUM_LIMBS;
use p3_keccak_air::NUM_ROUNDS;
use openvm_stark_backend::air_builders::sub::SubAirBuilder;
use openvm_circuit_primitives::utils::not;
use openvm_stark_backend::p3_air::AirBuilder;

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
        self.constrain_input_read(builder, local, &mut timestamp, &mem_oc.buffer_bytes_read_aux_cols);

        let keccak_f_air = KeccakAir {};
        let mut sub_builder =
            SubAirBuilder::<AB, KeccakAir, AB::Var>::new(builder, 0..NUM_KECCAK_PERM_COLS);
        keccak_f_air.eval(&mut sub_builder);

        // only active during the last round in each instruction 
        self.constrain_output_write(builder, local, &mut timestamp, &mem_oc.buffer_bytes_write_aux_cols);

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
        register_aux: &[MemoryReadAuxCols<AB::Var>; 1]
    ) {
        let instruction = local.instruction;
        let is_enabled = instruction.is_enabled;
        let is_first_round = local.inner.step_flags[0];
        let should_eval = is_enabled * is_first_round;
        builder.assert_bool(is_enabled);

        let reg_addr_sp = AB::F::ONE;
        let buffer_ptr = instruction.buffer_ptr;
        let buffer_data = instruction.buffer_limbs.map(Into::into);
        // 50 buffer reads and 50 buffer writes and 1 register read
        let timestamp_change = should_eval.clone() * AB::Expr::from_canonical_u32(1 + 50 + 50);

        // safety: it is safe to use timestamp instead of storing a separate instruction.start_timestamp
        // because the execution_bridge interaction is only enabled in the first row, so timestamp = start_timestamp
        // since there is no more timestamp increments 
        self.execution_bridge.execute_and_increment_pc(
            AB::Expr::from_canonical_usize(KeccakfOpcode::KECCAKF as usize + self.offset), 
            [
                buffer_ptr.into(),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
            ], 
            ExecutionState::new(instruction.pc, local.timestamp),
            timestamp_change,
        ).eval(builder, should_eval.clone());

        self.memory_bridge
            .read(MemoryAddress::new(reg_addr_sp, buffer_ptr), 
                buffer_data, 
                start_timestamp.clone(),
                &register_aux[0],
            )   
            .eval(builder, should_eval.clone());
        *start_timestamp += should_eval.clone();
        
        builder.assert_eq(
            instruction.buffer, 
            instruction.buffer_limbs[0] + 
            instruction.buffer_limbs[1] * AB::F::from_canonical_u32(1 << 8) + 
            instruction.buffer_limbs[2] * AB::F::from_canonical_u32(1 << 16) + 
            instruction.buffer_limbs[3] * AB::F::from_canonical_u32(1 << 24)
        );

        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        let need_range_check = [
            *instruction.buffer_limbs.last().unwrap(),
            *instruction.buffer_limbs.last().unwrap()
        ];
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, should_eval.clone());
        }
    }

    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_timestamp: &mut AB::Expr,
        buffer_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 200/4]
    ) {
        let is_enabled = local.instruction.is_enabled;
        let is_first_round = local.inner.step_flags[0];
        let should_read = is_enabled * is_first_round;
        
        const PREIMAGE_BYTES: usize = 25 * U64_LIMBS * 2;
        let local_preimage_bytes: [AB::Expr; PREIMAGE_BYTES] = std::array::from_fn(|byte_idx| {
            // `preimage` is represented as 5 * 5 * U64_LIMBS u16 limbs; each u16 limb is split into 2 bytes.
            let u16_idx = byte_idx / 2;
            let is_hi_byte = (byte_idx % 2) == 1;

            let i = u16_idx / U64_LIMBS;
            let limb = u16_idx % U64_LIMBS;
            let y = i / 5;
            let x = i % 5;

            let state_limb: AB::Expr = local.inner.preimage[y][x][limb].into();
            let hi: AB::Expr = local.preimage_state_hi[i * U64_LIMBS + limb].into();
            let lo: AB::Expr = state_limb - hi.clone() * AB::F::from_canonical_u64(1 << 8);

            if is_hi_byte { hi } else { lo }
        });

        for idx in 0..(PREIMAGE_BYTES / 4) {
            let read_chunk: [AB::Expr; 4] =
                std::array::from_fn(|j| local_preimage_bytes[4 * idx + j].clone());
            
            let ptr = local.instruction.buffer + AB::Expr::from_canonical_usize(idx * 4);

            self.memory_bridge.read(
                MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                read_chunk,
                start_timestamp.clone(),
                &buffer_bytes_read_aux_cols[idx]
            ).eval(builder, should_read.clone());

            *start_timestamp += should_read.clone();
        }
    }

    #[inline]
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        start_timestamp: &mut AB::Expr,
        buffer_bytes_write_aux_cols: &[MemoryWriteAuxCols<AB::Var, 4>; 200/4]
    ) {
        let is_enabled = local.instruction.is_enabled;
        let is_final_round = local.inner.step_flags[NUM_ROUNDS - 1];
        let should_write = is_enabled * is_final_round;
        
        const POSTIMAGE_BYTES: usize = 25 * U64_LIMBS * 2;
        let local_postimage_bytes: [AB::Expr; POSTIMAGE_BYTES] = std::array::from_fn(|byte_idx| {
            // `postimage` is represented as 5 * 5 * U64_LIMBS u16 limbs; each u16 limb is split into 2 bytes.
            let u16_idx = byte_idx / 2;
            let is_hi_byte = (byte_idx % 2) == 1;

            let i = u16_idx / U64_LIMBS;
            let limb = u16_idx % U64_LIMBS;
            let y = i / 5;
            let x = i % 5;

            let state_limb: AB::Expr = local.inner.a_prime_prime[y][x][limb].into();
            let hi: AB::Expr = local.postimage_state_hi[i * U64_LIMBS + limb].into();
            let lo: AB::Expr = state_limb - hi.clone() * AB::F::from_canonical_u64(1 << 8);

            if is_hi_byte { hi } else { lo }
        });

        for idx in 0..(POSTIMAGE_BYTES / 4) {
            let read_chunk: [AB::Expr; 4] =
                std::array::from_fn(|j| local_postimage_bytes[4 * idx + j].clone());
            
            let ptr = local.instruction.buffer + AB::Expr::from_canonical_usize(idx * 4);

            self.memory_bridge.write(
                MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                read_chunk,
                start_timestamp.clone(),
                &buffer_bytes_write_aux_cols[idx]
            ).eval(builder, should_write.clone());

            *start_timestamp += should_write.clone();
        }

    }

    // responsible for constraining everything that needs to be constrained between rows in the same instruction
    // but different keccakf rounds
    #[inline]
    pub fn constrain_rounds_transition<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
        next: &KeccakfVmCols<AB::Var>,
        timestamp: AB::Expr,
    ) {
        // todo: check if this needs to be only on when_transition 
        for idx in 0..100 {
            builder.assert_eq(local.preimage_state_hi[idx], next.preimage_state_hi[idx]);
            builder.assert_eq(local.postimage_state_hi[idx], next.postimage_state_hi[idx]);
            builder.assert_eq(local.instruction.pc, next.instruction.pc);
            builder.assert_eq(local.instruction.is_enabled, next.instruction.is_enabled);
            builder.assert_eq(local.instruction.buffer_ptr, next.instruction.buffer_ptr);
            builder.assert_eq(local.instruction.buffer, next.instruction.buffer);
            for limb in 0..4 {
                builder.assert_eq(local.instruction.buffer_limbs[limb], next.instruction.buffer_limbs[limb]);
            }
        }
        // safety: mem_oc does not need to be checked here because in the rows which is not first or last it is not used
        // and in the first or last rows, the mem_oc fields which is used is already constraiend by the interactions 
        let is_final_round = local.inner.step_flags[NUM_ROUNDS - 1];
        builder.when(not(is_final_round)).assert_eq(timestamp, next.timestamp);
    }

}


