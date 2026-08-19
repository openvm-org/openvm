use std::borrow::Borrow;

use openvm_circuit::{
    arch::{ExecutionBridge, ExecutionState, BLOCK_FE_WIDTH, MEMORY_BLOCK_BYTES},
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives::{
    bitwise_op_lookup::BitwiseOperationLookupBus, var_range::VariableRangeCheckerBus, ColumnsAir,
    StructReflection, StructReflectionHelper,
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{REGISTER_AS, REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::RevealOpcode;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};

use crate::adapters::{
    expand_to_block, pack_u8_pair, reg_byte_ptr_to_cell_ptr_limbs, PTR_U16_LIMBS, U16_BITS,
};

const REVEAL_TIMESTAMP_DELTA: usize = 4;

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct RevealCols<T> {
    /// Enables this row.
    pub is_valid: T,
    /// Execution state before this reveal.
    pub from_state: ExecutionState<T>,
    /// Byte pointer to the base-address register.
    pub base_ptr: T,
    /// Low 32 bits of the base address as u16 limbs.
    pub base_ptr_limbs: [T; PTR_U16_LIMBS],
    /// Witness for the base-register read.
    pub base_aux: MemoryReadAuxCols<T>,
    /// Byte pointer to the source-value register.
    pub src_ptr: T,
    /// Byte decomposition of the source register value.
    pub src_bytes: [T; REGISTER_NUM_LIMBS],
    /// Witness for the source-register read.
    pub src_aux: MemoryReadAuxCols<T>,
    /// Low 16 bits of the signed address offset.
    pub imm: T,
    /// Sign bit of the address offset.
    pub imm_sign: T,
    /// Low u16 limb of the aligned reveal address.
    pub dst_ptr_low_limb: T,
    /// Witness for the public-values write.
    pub write_aux: [MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>; 2],
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(RevealCols<u8>)]
pub struct RevealAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
    pub range_bus: VariableRangeCheckerBus,
    pub bitwise_bus: BitwiseOperationLookupBus,
    pub pointer_max_bits: usize,
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

        // Read the low 32-bit base address from its register.
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.base_ptr),
                ),
                expand_to_block(&cols.base_ptr_limbs),
                timestamp.clone(),
                &cols.base_aux,
            )
            .eval(builder, is_valid.clone());

        // Add the signed immediate across the two u16 pointer limbs.
        let inv_u16_base = AB::F::from_u32(1 << U16_BITS).inverse();
        let low_carry = (cols.base_ptr_limbs[0] + cols.imm - cols.dst_ptr_low_limb) * inv_u16_base;
        builder.assert_bool(low_carry.clone());
        let dst_ptr_high_limb = cols.base_ptr_limbs[1] + low_carry - cols.imm_sign;

        // Enforce 8-byte alignment and the configured pointer bound.
        let block_bytes = AB::F::from_usize(MEMORY_BLOCK_BYTES);
        self.range_bus
            .range_check(cols.dst_ptr_low_limb * block_bytes.inverse(), U16_BITS - 3)
            .eval(builder, is_valid.clone());
        self.range_bus
            .range_check(dst_ptr_high_limb.clone(), self.pointer_max_bits - U16_BITS)
            .eval(builder, is_valid.clone());

        // Compose the source-register bus value directly from its byte limbs.
        let src_cells: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|i| {
            pack_u8_pair(
                cols.src_bytes[2 * i].into(),
                cols.src_bytes[2 * i + 1].into(),
            )
        });
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(cols.src_ptr),
                ),
                src_cells,
                timestamp.clone() + AB::Expr::ONE,
                &cols.src_aux,
            )
            .eval(builder, is_valid.clone());
        for bytes in cols.src_bytes.chunks_exact(2) {
            self.bitwise_bus
                .send_range(bytes[0], bytes[1])
                .eval(builder, is_valid.clone());
        }

        // One RV64 register expands to two four-byte public-values writes.
        for (chunk_idx, (bytes, aux)) in cols
            .src_bytes
            .chunks_exact(BLOCK_FE_WIDTH)
            .zip(&cols.write_aux)
            .enumerate()
        {
            let values: [AB::Expr; BLOCK_FE_WIDTH] = std::array::from_fn(|lane| bytes[lane].into());
            // Public values are byte-celled, so divide the byte pointer by four cells per block.
            let block_index = cols.dst_ptr_low_limb * AB::F::from_usize(BLOCK_FE_WIDTH).inverse()
                + dst_ptr_high_limb.clone()
                    * AB::F::from_u32(1 << (U16_BITS - BLOCK_FE_WIDTH.ilog2() as usize))
                + AB::F::from_usize(chunk_idx);
            self.memory_bridge
                .write(
                    MemoryAddress::new(AB::F::from_u32(PUBLIC_VALUES_AS), block_index),
                    values,
                    timestamp.clone() + AB::Expr::from_usize(2 + chunk_idx),
                    aux,
                )
                .eval(builder, is_valid.clone());
        }

        // Bind the row to the dedicated opcode and its four memory events.
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
