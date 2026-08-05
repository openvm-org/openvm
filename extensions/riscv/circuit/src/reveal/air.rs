use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    var_range::VariableRangeCheckerBus, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP, riscv::REGISTER_AS, LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::adapters::{byte_ptr_to_u16_ptr, expand_to_block, PTR_U16_LIMBS, U16_BITS};

const REVEAL_TIMESTAMP_DELTA: usize = 3;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RevealCols<T> {
    pub is_valid: T,
    pub from_state: ExecutionState<T>,
    pub base_ptr: T,
    pub base_data: [T; PTR_U16_LIMBS],
    pub base_aux: MemoryReadAuxCols<T>,
    pub src_ptr: T,
    pub src_data: [T; BLOCK_FE_WIDTH],
    pub src_aux: MemoryReadAuxCols<T>,
    pub imm: T,
    pub imm_sign: T,
    pub reveal_ptr_low_limb: T,
    pub write_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[derive(Clone, Copy, Debug, ColumnsAir)]
#[columns_via(RevealCols<u8>)]
pub struct RevealAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub pointer_max_bits: usize,
}

impl RevealAir {
    pub fn new(
        execution_bridge: ExecutionBridge,
        memory_bridge: MemoryBridge,
        range_bus: VariableRangeCheckerBus,
        pointer_max_bits: usize,
    ) -> Self {
        Self {
            execution_bridge,
            memory_bridge,
            range_bus,
            pointer_max_bits,
        }
    }
}

impl<F: Field> BaseAir<F> for RevealAir {
    fn width(&self) -> usize {
        RevealCols::<F>::width()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for RevealAir {}
impl<F: Field> PartitionedBaseAir<F> for RevealAir {}

impl<AB: InteractionBuilder> Air<AB> for RevealAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("reveal AIR requires a local row");
        let cols: &RevealCols<AB::Var> = (*local).borrow();
        let is_valid: AB::Expr = cols.is_valid.into();
        let timestamp: AB::Expr = cols.from_state.timestamp.into();

        builder.assert_bool(cols.is_valid);
        builder.assert_bool(cols.imm_sign);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(cols.base_ptr),
                ),
                expand_to_block(&cols.base_data),
                timestamp.clone(),
                &cols.base_aux,
            )
            .eval(builder, is_valid.clone());

        let inv_u16_base = AB::F::from_u32(1 << U16_BITS).inverse();
        let low_carry = (cols.base_data[0] + cols.imm - cols.reveal_ptr_low_limb) * inv_u16_base;
        builder.assert_bool(low_carry.clone());
        let reveal_ptr_high_limb = cols.base_data[1] + low_carry - cols.imm_sign;

        let block_bytes = AB::F::from_usize(MEMORY_BLOCK_BYTES);
        self.range_bus
            .range_check(
                cols.reveal_ptr_low_limb * block_bytes.inverse(),
                U16_BITS - 3,
            )
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(
                reveal_ptr_high_limb.clone(),
                self.pointer_max_bits - U16_BITS,
            )
            .eval(builder, is_valid.clone());
        let reveal_ptr =
            cols.reveal_ptr_low_limb + reveal_ptr_high_limb * AB::F::from_u32(1 << U16_BITS);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(cols.src_ptr),
                ),
                cols.src_data.map(Into::into),
                timestamp.clone() + AB::Expr::ONE,
                &cols.src_aux,
            )
            .eval(builder, is_valid.clone());
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(PUBLIC_VALUES_AS),
                    byte_ptr_to_u16_ptr::<AB>(reveal_ptr),
                ),
                cols.src_data.map(Into::into),
                timestamp.clone() + AB::Expr::TWO,
                &cols.write_aux,
            )
            .eval(builder, is_valid.clone());

        self.execution_bridge
            .execute(
                AB::Expr::from_usize(RevealOpcode::REVEAL.global_opcode().as_usize()),
                [
                    cols.src_ptr.into(),
                    cols.base_ptr.into(),
                    cols.imm.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(PUBLIC_VALUES_AS),
                    is_valid.clone(),
                    cols.imm_sign.into(),
                ],
                cols.from_state,
                ExecutionState {
                    pc: cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP),
                    timestamp: timestamp + AB::F::from_usize(REVEAL_TIMESTAMP_DELTA),
                },
            )
            .eval(builder, is_valid);
    }
}
