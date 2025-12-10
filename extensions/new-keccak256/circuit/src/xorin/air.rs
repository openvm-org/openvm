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
use openvm_new_keccak256_transpiler::Rv32NewKeccakOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::FieldAlgebra,
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};

use crate::xorin::columns::{XorinVmCols, NUM_XORIN_VM_COLS};

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
        let local: &XorinVmCols<AB::Var> = (*local).borrow();

        let mem = &local.mem_oc;

        let start_read_timestamp = self.eval_instruction(builder, local, &mem.register_aux_cols);

        let start_write_timestamp = self.constrain_input_read(builder, local, start_read_timestamp, &mem.input_bytes_read_aux_cols, &mem.buffer_bytes_read_aux_cols);

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
        let should_receive = local.instruction.is_enabled;

        let [buffer_ptr, input_ptr, len_ptr] = [
            instruction.buffer_ptr,
            instruction.input_ptr,
            instruction.len_ptr,
        ];

        let reg_addr_sp = AB::F::ONE;

        // todo: fill this in
        let timestamp_change =
            local.instruction.len + local.instruction.len + local.instruction.len;

        self.execution_bridge
            .execute_and_increment_pc(
                AB::Expr::from_canonical_usize(Rv32NewKeccakOpcode::XORIN as usize + self.offset),
                [
                    buffer_ptr.into(),
                    input_ptr.into(),
                    len_ptr.into(),
                    reg_addr_sp.into(),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change,
            )
            .eval(builder, should_receive);

        let mut timestamp: AB::Expr = instruction.start_timestamp.into();

        let buffer_data = instruction.buffer_limbs.map(Into::into);
        let input_data = instruction.input_limbs.map(Into::into);
        let len_data = instruction.input_limbs.map(Into::into);

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
                .eval(builder, should_receive.clone());

            timestamp += AB::Expr::ONE;
        }

        // todo: range check the buffer limbs, input limbs and length limbs
        let need_range_check = [
            *instruction.buffer_limbs.last().unwrap(),
            *instruction.input_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
            *instruction.len_limbs.last().unwrap(),
        ];

        // todo: recheck if this safety needs to be fixed
        // SAFETY: this approach does not work when self.ptr_max_bits < 24
        let limb_shift = AB::F::from_canonical_usize(
            1 << (RV32_CELL_BITS * RV32_REGISTER_NUM_LIMBS - self.ptr_max_bits),
        );
        for pair in need_range_check.chunks_exact(2) {
            self.bitwise_lookup_bus
                .send_range(pair[0] * limb_shift, pair[1] * limb_shift)
                .eval(builder, should_receive.clone());
        }

        timestamp
    }

    // Increases timestamp by 2 * 34 = 68
    #[inline]
    pub fn constrain_input_read<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_read_timestamp: AB::Expr,
        input_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 34],
        buffer_bytes_read_aux_cols: &[MemoryReadAuxCols<AB::Var>; 34],
    ) -> AB::Expr {
        let mut timestamp = start_read_timestamp;

        // Constrain that is_padding_bytes is boolean
        for is_padding in local.sponge.is_padding_bytes {
            builder.assert_bool(is_padding);
        }

        // Constrain read of input_bytes
        // Timestamp increases by exactly (136/4) = 34
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.input_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            input_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.input + AB::F::from_canonical_usize(i * 4);

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
                .eval(builder, not(is_padding));

            timestamp += AB::Expr::ONE;
        }

        // Constrain read of buffer bytes
        // Timestamp increases by exactly (136/4) = 34
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.preimage_buffer_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            buffer_bytes_read_aux_cols
        )
        .enumerate()
        {
            let ptr = local.instruction.buffer + AB::F::from_canonical_usize(i * 4);

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
                .eval(builder, not(is_padding));

            timestamp += AB::Expr::ONE;
        }

        timestamp
    }

    pub fn constrain_xor<AB: InteractionBuilder>(
        &self,
        builder: &mut AB, 
        local: &XorinVmCols<AB::Var>
    ) {
        let buffer_bytes = local.sponge.preimage_buffer_bytes;
        let input_bytes = local.sponge.input_bytes;
        let result_bytes = local.sponge.postimage_buffer_bytes;
        let padding_bytes = local.sponge.is_padding_bytes;

        for (x, y, x_xor_y, is_padding) in izip!(
            buffer_bytes,
            input_bytes,
            result_bytes,
            padding_bytes
        )
        {
            self.bitwise_lookup_bus.send_xor(x, y, x_xor_y).eval(builder, is_padding);
        }
    }

    // Increases timestamp by 34
    pub fn constrain_output_write<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        start_write_timestamp: AB::Expr,
        mem_aux: &[MemoryWriteAuxCols<AB::Var, 4>; 34]
    ) {
        let mut timestamp = start_write_timestamp;
        // Constrain write of buffer bytes
        for (i, (input, is_padding, mem_aux)) in izip!(
            local.sponge.postimage_buffer_bytes.chunks_exact(4),
            local.sponge.is_padding_bytes,
            mem_aux
        )
        .enumerate()
        {
            let ptr = local.instruction.buffer + AB::F::from_canonical_usize(i * 4);

            self.memory_bridge
                .write(
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
                .eval(builder, not(is_padding));

            timestamp += AB::Expr::ONE;
        }
    }
}
