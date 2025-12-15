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
use crate::keccakf::columns::{KeccakfVmCols, NUM_KECCAKF_VM_COLS};
use openvm_new_keccak256_transpiler::KeccakfOpcode;
use openvm_instructions::riscv::RV32_REGISTER_AS;
use openvm_stark_backend::interaction::PermutationCheckBus;

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct KeccakfVmAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub bitwise_lookup_bus: BitwiseOperationLookupBus,
    pub ptr_max_bits: usize,
    pub(super) offset: usize,
    pub keccak_bus: PermutationCheckBus,
}

pub struct KeccakfWrapperAir {
    pub keccak_bus: PermutationCheckBus
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

        let local = main.row_slice(0);
        let local: &KeccakfVmCols<_> = (*local).borrow();

        let mut timestamp: AB::Expr = local.instruction.start_timestamp.into();
        let mem_oc = &local.mem_oc;

        // increases timestamp by 1
        self.eval_instruction(builder, local, &mut timestamp, &mem_oc.register_aux_cols);
        // increases timestamp by 50
        self.constrain_input_read(builder, local, &mut timestamp, &mem_oc.buffer_bytes_read_aux_cols);

        // todo: call communicate with keccak bus here
        self.communicate_with_keccakbus(builder, local);

        // increases timestamp by 50
        self.constrain_output_write(builder, local, &mut timestamp, &mem_oc.buffer_bytes_write_aux_cols);

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
        builder.assert_bool(is_enabled);
        let reg_addr_sp = AB::F::ONE;

        let buffer_ptr = instruction.buffer_ptr;
        let buffer_data = instruction.buffer_limbs.map(Into::into);
        // 50 buffer reads and 50 buffer writes and 1 register read
        let timestamp_change = AB::Expr::from_canonical_u32(2 * 50 + 1);

        // todo: check if the below operands are correct. do i need to make the second and third operands be
        // zero instead and make the 4th and 5th be the current 2nd and 3rd?
        self.execution_bridge.execute_and_increment_pc(
            AB::Expr::from_canonical_usize(KeccakfOpcode::KECCAKF as usize + self.offset), 
            [
                buffer_ptr.into(),
                AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
                AB::Expr::from_canonical_u32(RV32_MEMORY_AS),
            ], 
            ExecutionState::new(instruction.pc, instruction.start_timestamp),
            timestamp_change,
        ).eval(builder, is_enabled);

        self.memory_bridge
            .read(MemoryAddress::new(reg_addr_sp, buffer_ptr), 
                buffer_data, 
                start_timestamp.clone(),
                &register_aux[0],
            )   
            .eval(builder, is_enabled);
        *start_timestamp += AB::Expr::ONE;
        
        builder.assert_eq(instruction.buffer, instruction.buffer_limbs[0] + instruction.buffer_limbs[1] * AB::F::from_canonical_u32(1 << 8) + instruction.buffer_limbs[2] * AB::F::from_canonical_u32(1 << 16) + instruction.buffer_limbs[3] * AB::F::from_canonical_u32(1 << 24));
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
        for idx in 0..50 {
            let chunk: [AB::Var; 4] = local.sponge.preimage_buffer_bytes[4 * idx .. 4 * idx + 4].try_into().unwrap();
            self.memory_bridge
                .read(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), local.instruction.buffer + AB::F::from_canonical_usize(idx * 4)),
                    chunk,
                    start_timestamp.clone(),
                    &buffer_bytes_read_aux_cols[idx]
                ).eval(builder, is_enabled);

            *start_timestamp += AB::Expr::ONE;
        }
    }

    #[inline]
    pub fn communicate_with_keccakbus<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &KeccakfVmCols<AB::Var>,
    ) {
        let is_enabled = local.instruction.is_enabled;
        self.keccak_bus.send(
            builder,
            local
                .sponge
                .preimage_buffer_bytes
                .into_iter()
                .map(Into::into)
                .chain(once(local.request_id.into())),
            is_enabled,
        );
        self.keccak_bus.receive(
            builder, 
            local
                .sponge
                .postimage_buffer_bytes
                .into_iter()
                .map(Into::into)
                .chain(once(local.request_id.into())),
            is_enabled
        );
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
        for idx in 0..50 {
            let chunk: [AB::Var; 4] = local.sponge.postimage_buffer_bytes[4 * idx .. 4 * idx + 4].try_into().unwrap();
            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::Expr::from_canonical_u32(RV32_MEMORY_AS), local.instruction.buffer + AB::F::from_canonical_usize(idx * 4)),
                    chunk,
                    start_timestamp.clone(),
                    &buffer_bytes_write_aux_cols[idx]
                )
                .eval(builder, is_enabled);

            *start_timestamp += AB::Expr::ONE;
        }
    }

}