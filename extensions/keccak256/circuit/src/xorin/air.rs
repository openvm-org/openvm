use std::borrow::Borrow;

use itertools::izip;
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, DEFAULT_BLOCK_SIZE},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    utils::{compose, not},
};
use openvm_instructions::riscv::{
    RV64_CELL_BITS, RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS, RV64_WORD_NUM_LIMBS,
};
use openvm_keccak256_transpiler::XorinOpcode;
use openvm_riscv_circuit::adapters::expand_to_rv64_register;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::{
    xorin::columns::{XorinVmCols, NUM_XORIN_VM_COLS},
    KECCAK_RATE_MEM_OPS,
};

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct XorinVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    /// Bus to send 8-bit XOR requests to.
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    /// Maximum number of bits allowed for an address pointer
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
}

impl<F> BaseAirWithPublicValues<F> for XorinVmAir {}
impl<F> PartitionedBaseAir<F> for XorinVmAir {}
impl<F> BaseAir<F> for XorinVmAir {
    fn width(&self) -> usize {
        NUM_XORIN_VM_COLS
    }
}

impl<AB: InteractionBuilder> Air<AB> for XorinVmAir {
    // Increases timestamp by 3 + 3*17 = 54
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0).unwrap();
        let local: &XorinVmCols<_> = (*local).borrow();

        let mem = &local.mem_oc;

        let start_read_timestamp = self.eval_instruction(builder, local, &mem.register_aux_cols);

        let start_write_timestamp = self.constrain_input_read(
            builder,
            local,
            start_read_timestamp,
            &mem.input_bytes_read_aux_cols,
            &mem.buffer_bytes_read_aux_cols,
        );

        self.constrain_xor(builder, local);

        self.constrain_output_write(
            builder,
            local,
            start_write_timestamp,
            &mem.buffer_bytes_write_aux_cols,
        );
    }
}

impl XorinVmAir {
    // Increases timestamp by 3
    #[inline]
    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        register_aux: &[MemoryReadAuxCols<AB::Var>; 3],
    ) -> AB::Expr {
        // returns start_read_timestamp
        let instruction = local.instruction;
        let is_enabled = local.instruction.is_enabled;
        builder.assert_bool(is_enabled);

        let [buffer_reg_ptr, input_reg_ptr, len_reg_ptr] = [
            instruction.buffer_reg_ptr,
            instruction.input_reg_ptr,
            instruction.len_reg_ptr,
        ];

        let mut timestamp_change = AB::Expr::from_u32(3);
        let mut not_padding_sum = AB::Expr::ZERO;
        let is_padding_bytes = local.sponge.is_padding_bytes;

        // Check that is_padding_bytes is of the form 0...01...1
        for (i, &is_padding) in is_padding_bytes.iter().enumerate() {
            builder.assert_bool(is_padding);
            not_padding_sum += not(is_padding);
            if i > 0 {
                builder
                    .when(is_enabled)
                    .assert_bool(is_padding - is_padding_bytes[i - 1]);
            }
            // Each 8-byte memory block has 3 ops (buffer read + input read + buffer write).
            timestamp_change += AB::Expr::from_u32(3) * not(is_padding);
        }

        not_padding_sum *= AB::Expr::from_usize(DEFAULT_BLOCK_SIZE);
        builder
            .when(is_enabled)
            .assert_eq(not_padding_sum, instruction.len);

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_usize(XorinOpcode::XORIN as usize + self.offset),
                [
                    buffer_reg_ptr.into(),
                    input_reg_ptr.into(),
                    len_reg_ptr.into(),
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::from_u32(RV64_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change,
            )
            .eval(builder, is_enabled);

        let mut timestamp: AB::Expr = instruction.start_timestamp.into();

        // Build full 8-element data arrays with upper 4 limbs hardcoded to zero
        let buffer_ptr_limbs: [AB::Expr; RV64_REGISTER_NUM_LIMBS] =
            expand_to_rv64_register(&instruction.buffer_ptr_limbs);
        let input_ptr_limbs: [AB::Expr; RV64_REGISTER_NUM_LIMBS] =
            expand_to_rv64_register(&instruction.input_ptr_limbs);
        let len_limbs: [AB::Expr; RV64_REGISTER_NUM_LIMBS] =
            expand_to_rv64_register(&[instruction.len_limb]);

        // Increases timestamp by 3
        for (ptr, value, aux) in izip!(
            [buffer_reg_ptr, input_reg_ptr, len_reg_ptr],
            [buffer_ptr_limbs, input_ptr_limbs, len_limbs],
            register_aux
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_REGISTER_AS), ptr),
                    value,
                    timestamp.clone(),
                    aux,
                )
                .eval(builder, is_enabled);

            timestamp += AB::Expr::ONE;
        }

        // SAFETY: this approach only works when self.ptr_max_bits >= RV64_CELL_BITS *
        // (RV64_WORD_NUM_LIMBS - 1) because we are only range checking the MSB of the lower
        // address bytes
        let need_range_check = [
            instruction.buffer_ptr_limbs[RV64_WORD_NUM_LIMBS - 1],
            instruction.input_ptr_limbs[RV64_WORD_NUM_LIMBS - 1],
        ];

        let limb_shift =
            AB::F::from_usize(1 << (RV64_CELL_BITS * RV64_WORD_NUM_LIMBS - self.ptr_max_bits));
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, is_enabled);
        }

        builder.assert_eq(
            instruction.buffer_ptr,
            compose(&instruction.buffer_ptr_limbs[..], RV64_CELL_BITS),
        );

        builder.assert_eq(
            instruction.input_ptr,
            compose(&instruction.input_ptr_limbs[..], RV64_CELL_BITS),
        );

        builder.assert_eq(instruction.len, instruction.len_limb);

        timestamp
    }

    // Increases timestamp by <= 2 * 17 = 34
    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
        input_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; KECCAK_RATE_MEM_OPS],
        buffer_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; KECCAK_RATE_MEM_OPS],
    ) -> AB::Expr {
        let is_enabled = local.instruction.is_enabled;
        let mut timestamp = start_read_timestamp;

        // Constrain read of buffer bytes
        // Timestamp increases by <= (136/8) = 17
        for (i, (input, mem_aux)) in izip!(
            local
                .sponge
                .preimage_buffer_bytes
                .chunks_exact(DEFAULT_BLOCK_SIZE),
            buffer_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.buffer_ptr + AB::F::from_usize(i * DEFAULT_BLOCK_SIZE);
            let is_padding = local.sponge.is_padding_bytes[i];
            let should_read = is_enabled * not(is_padding);

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_MEMORY_AS), ptr),
                    [
                        input[0].into(),
                        input[1].into(),
                        input[2].into(),
                        input[3].into(),
                        input[4].into(),
                        input[5].into(),
                        input[6].into(),
                        input[7].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_read.clone());

            timestamp += not(is_padding);
        }

        // Constrain read of input_bytes
        // Timestamp increases by at most (136/8) = 17
        for (i, (input, mem_aux)) in izip!(
            local.sponge.input_bytes.chunks_exact(DEFAULT_BLOCK_SIZE),
            input_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.input_ptr + AB::F::from_usize(i * DEFAULT_BLOCK_SIZE);
            let is_padding = local.sponge.is_padding_bytes[i];
            let should_read = is_enabled * not(is_padding);

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_MEMORY_AS), ptr),
                    [
                        input[0].into(),
                        input[1].into(),
                        input[2].into(),
                        input[3].into(),
                        input[4].into(),
                        input[5].into(),
                        input[6].into(),
                        input[7].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_read.clone());

            timestamp += not(is_padding);
        }

        timestamp
    }

    #[inline]
    pub fn constrain_xor<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
    ) {
        let buffer_bytes = local.sponge.preimage_buffer_bytes;
        let input_bytes = local.sponge.input_bytes;
        let result_bytes = local.sponge.postimage_buffer_bytes;
        let padding_bytes = local.sponge.is_padding_bytes;
        let is_enabled = local.instruction.is_enabled;

        for (x_chunks, y_chunks, x_xor_y_chunks, is_padding) in izip!(
            buffer_bytes.chunks_exact(DEFAULT_BLOCK_SIZE),
            input_bytes.chunks_exact(DEFAULT_BLOCK_SIZE),
            result_bytes.chunks_exact(DEFAULT_BLOCK_SIZE),
            padding_bytes
        ) {
            let should_send = is_enabled * not(is_padding);
            for (x, y, x_xor_y) in izip!(x_chunks, y_chunks, x_xor_y_chunks) {
                self.bitwise_lookup_bus
                    .send_xor(*x, *y, *x_xor_y)
                    .eval(builder, should_send.clone());
            }
        }
    }

    #[inline]
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
        mem_aux: &[MemoryWriteAuxCols<AB::Var, DEFAULT_BLOCK_SIZE>; KECCAK_RATE_MEM_OPS],
    ) {
        let mut timestamp = start_write_timestamp;
        let is_enabled = local.instruction.is_enabled;

        // Constrain write of buffer bytes
        for (i, (output, mem_aux)) in izip!(
            local
                .sponge
                .postimage_buffer_bytes
                .chunks_exact(DEFAULT_BLOCK_SIZE),
            mem_aux
        )
        .enumerate()
        {
            let is_padding = local.sponge.is_padding_bytes[i];
            let should_write = is_enabled * not(is_padding);
            let ptr = local.instruction.buffer_ptr + AB::F::from_usize(i * DEFAULT_BLOCK_SIZE);

            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::Expr::from_u32(RV64_MEMORY_AS), ptr),
                    [
                        output[0].into(),
                        output[1].into(),
                        output[2].into(),
                        output[3].into(),
                        output[4].into(),
                        output[5].into(),
                        output[6].into(),
                        output[7].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_write);

            timestamp += not(is_padding);
        }
    }
}
