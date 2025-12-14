use std::{array::from_fn, borrow::Borrow, iter::zip};

use itertools::{izip, Itertools};
use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus,
    utils::{not, select},
};
use openvm_instructions::riscv::{RV32_CELL_BITS, RV32_MEMORY_AS, RV32_REGISTER_NUM_LIMBS};
use openvm_new_keccak256_transpiler::XorinOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::xorin::columns::{XorinVmCols, NUM_XORIN_VM_COLS};
use openvm_stark_backend::p3_air::AirBuilder;
use openvm_instructions::riscv::RV32_REGISTER_AS;


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
    // Increases timestamp by 105
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();

        let local = main.row_slice(0);
        let local: &XorinVmCols<_> = (*local).borrow();

        let mem = &local.mem_oc;

        let start_read_timestamp = self.eval_instruction(builder, local, &mem.register_aux_cols);

        let start_write_timestamp = self.constrain_input_read(builder, local, start_read_timestamp, &mem.input_bytes_read_aux_cols, &mem.buffer_bytes_read_aux_cols);

        self.constrain_xor(builder, local);

        self.constrain_output_write(builder, local, start_write_timestamp, &mem.buffer_bytes_write_aux_cols);
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

        let [buffer_ptr, input_ptr, len_ptr] = [
            instruction.buffer_ptr,
            instruction.input_ptr,
            instruction.len_ptr,
        ];

        let reg_addr_sp = AB::F::ONE;
        let mut timestamp_change = AB::Expr::from_canonical_u32(3);

        for is_padding in local.sponge.is_padding_bytes {
            timestamp_change += not(is_padding);
            timestamp_change += not(is_padding);
            timestamp_change += not(is_padding);
        }

        let mut not_padding_sum = AB::Expr::ZERO;
        for is_padding in local.sponge.is_padding_bytes {
            not_padding_sum += not(is_padding);
        }
        not_padding_sum *= AB::Expr::from_canonical_u32(4);
        builder.when(is_enabled).assert_eq(not_padding_sum, instruction.len);
        // check that is_padding_bytes is of the form 0...0111...1
        for i in 0..33 {
            builder.when(is_enabled).assert_bool(local.sponge.is_padding_bytes[i+1] - local.sponge.is_padding_bytes[i]);
        }

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(XorinOpcode::XORIN as usize + self.offset),
                [
                    buffer_ptr.into(),
                    input_ptr.into(),
                    len_ptr.into(),
                    AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change,
            )
            .eval(builder, is_enabled);

        let mut timestamp: AB::Expr = instruction.start_timestamp.into();

        let buffer_data = instruction.buffer_limbs.map(Into::into);
        let input_data = instruction.input_limbs.map(Into::into);
        let len_data = instruction.len_limbs.map(Into::into);

        // Increases timestamp by 3
        for (ptr, value, aux) in izip!(
            [buffer_ptr, input_ptr, len_ptr],
            [buffer_data, input_data, len_data],
            register_aux
        ) {
            self.memory_bridge
                .read(
                    MemoryAddress::new(reg_addr_sp, ptr),
                    value,
                    timestamp.clone(),
                    aux,
                )
                .eval(builder, is_enabled);

            timestamp += AB::Expr::ONE;
        }
        
        // SAFETY: this approach only works when self.ptr_max_bits >= 24
        // because we are only range checking the last limb 
        let need_range_check = [
            *instruction.buffer_limbs.last().unwrap(),
            *instruction.input_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
        ];
        
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, is_enabled);
        }

        builder.assert_eq(instruction.buffer, instruction.buffer_limbs[0] + instruction.buffer_limbs[1] * AB::F::from_canonical_u32(1 << 8) + instruction.buffer_limbs[2] * AB::F::from_canonical_u32(1 << 16) + instruction.buffer_limbs[3] * AB::F::from_canonical_u32(1 << 24));

        builder.assert_eq(instruction.input, instruction.input_limbs[0] + instruction.input_limbs[1] * AB::F::from_canonical_u32(1 << 8) + instruction.input_limbs[2] * AB::F::from_canonical_u32(1 << 16) + instruction.input_limbs[3] * AB::F::from_canonical_u32(1 << 24));

        builder.assert_eq(instruction.len, instruction.len_limbs[0] + instruction.len_limbs[1] * AB::F::from_canonical_u32(1 << 8) + instruction.len_limbs[2] * AB::F::from_canonical_u32(1 << 16) + instruction.len_limbs[3] * AB::F::from_canonical_u32(1 << 24));

        timestamp
    }

    // Increases timestamp by <= 2 * 34 = 68
    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
        input_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 34],
        buffer_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 34],
    ) -> AB::Expr {
        let is_enabled = local.instruction.is_enabled;
        let mut timestamp = start_read_timestamp;

        // Constrain read of buffer bytes
        // Timestamp increases by <= (136/4) = 34
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.preimage_buffer_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            buffer_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.buffer + AB::F::from_canonical_usize(i * 4);
            let should_read = is_enabled * not(is_padding);

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    [
                        input[0].into(),
                        input[1].into(),
                        input[2].into(),
                        input[3].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_read);

            timestamp += not(is_padding);
        }

        // Constrain read of input_bytes
        // Timestamp increases by at most (136/4) = 34
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.input_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            input_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.input + AB::F::from_canonical_usize(i * 4);
            let should_read = is_enabled * not(is_padding);

            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    [
                        input[0].into(),
                        input[1].into(),
                        input[2].into(),
                        input[3].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_read);

            timestamp += not(is_padding);
        }

        timestamp
    }

    #[inline]
    pub fn constrain_xor<AB: InteractionBuilder>(
        &self,
        builder: &mut AB, 
        local: &XorinVmCols<AB::Var>
    ) {
        let buffer_bytes = local.sponge.preimage_buffer_bytes;
        let input_bytes = local.sponge.input_bytes;
        let result_bytes = local.sponge.postimage_buffer_bytes;
        let padding_bytes = local.sponge.is_padding_bytes;
        let is_enabled = local.instruction.is_enabled;

        for (x_chunks, y_chunks, x_xor_y_chunks, is_padding) in izip!(
            buffer_bytes.chunks_exact(4),
            input_bytes.chunks_exact(4),
            result_bytes.chunks_exact(4),
            padding_bytes
        )
        {
            let should_send = is_enabled * not(is_padding);
            for (x, y, x_xor_y) in izip!(x_chunks, y_chunks, x_xor_y_chunks) {
                self.bitwise_lookup_bus.send_xor(*x, *y, *x_xor_y).eval(builder, should_send.clone());
            }
        }
    }

    #[inline]
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
        mem_aux: &[MemoryWriteAuxCols<AB::Var, 4>; 34]
    ) {
        let mut timestamp = start_write_timestamp;
        let is_enabled = local.instruction.is_enabled;
        
        // Constrain write of buffer bytes
        for (i, (output, is_padding, mem_aux)) in izip!(
            local.sponge.postimage_buffer_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            mem_aux
        )
        .enumerate()
        {
            let should_write = is_enabled * not(is_padding);
            let ptr = local.instruction.buffer + AB::F::from_canonical_usize(i * 4);

            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), ptr),
                    [
                        output[0].into(),
                        output[1].into(),
                        output[2].into(),
                        output[3].into(),
                    ],
                    timestamp.clone(),
                    mem_aux,
                )
                .eval(builder, should_write);

            timestamp += not(is_padding);
        }
    }
}
