use std::{array::from_fn, borrow::Borrow, iter::zip};
use openvm_circuit::arch::ExecutionBridge;
use openvm_circuit::system::memory::offline_checker::MemoryBridge;
use openvm_circuit_primitives::bitwise_op_lookup::BitwiseOperationLookupBus;
use openvm_instructions::riscv::RV32_MEMORY_AS;
use openvm_stark_backend::p3_matrix::Matrix;
use openvm_stark_backend::rap::BaseAirWithPublicValues;
use openvm_stark_backend::rap::PartitionedBaseAir;
use openvm_stark_backend::p3_air::BaseAir;
use openvm_stark_backend::p3_air::Air;
use openvm_stark_backend::interaction::InteractionBuilder;

use crate::xorin::columns::NUM_XORIN_VM_COLS;
use crate::xorin::columns::XorinVmCols;
use openvm_circuit::system::memory::offline_checker::MemoryReadAuxCols;
use openvm_stark_backend::p3_field::FieldAlgebra;

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
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (main.row_slice(0), main.row_slice(1));

        let local: &XorinVmCols<AB::Var> = (*local).borrow();
        let next: &XorinVmCols<AB::Var> = (*next).borrow();


    }
}

impl XorinVmAir {
    #[inline]
    pub fn eval_instruction<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &XorinVmCols<AB::Var>,
        register_aux: &[MemoryReadAuxCols<AB::Var>; 3],
    ) -> AB::Expr { // returns start_read_timestamp
        let instruction = local.instruction;
        let should_receive = local.instruction.is_enabled;

        let [buffer_ptr, input_ptr, len_ptr] = [
            instruction.buffer_ptr,
            instruction.input_ptr,
            instruction.len_ptr
        ];

        let reg_addr_sp = AB::F::ONE;
        
        // todo: fill this in
        let timestamp_change = ;

        self.execution_bridge
            .execute_and_increment_pc(
                ::Expr::from_canonical_usize(Rv32NewKeccakOpcode::KECCAK256 as usize + self.offset),
                [
                    buffer_ptr.into(),
                    input_ptr.into(),
                    len_ptr.into(),
                    reg_addr_sp.into(),
                    AB::Expr::from_canonical_u32(RV32_MEMORY_AS)
                ],
                ExecutionState::new(instruction.pc, instruction.start_timestamp),
                timestamp_change
            )
            .eval(builder, should_receive.clone());

        let mut timestamp = AB::Expr = instruction.start_timestamp.into();
        
        let buffer_data = instruction.buffer_limbs.map(Into::into);
        let input_data = instruction.input_limbs.map(Into::into);
        let len_data = instruction.input_limbs.map(Into::into);

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
}