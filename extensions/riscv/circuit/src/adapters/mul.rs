use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, Postflight, PostflightError, PostflightStep, VmAdapterAir,
        BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{pack_u8_block, MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{pc_to_limbs, DEFAULT_PC_STEP},
    riscv::REGISTER_AS,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{ReplayComputation, ReplayResult, REGISTER_NUM_LIMBS};
use crate::adapters::{
    bytes_to_u16_block, checked_register_u16_pointer, reg_byte_ptr_to_cell_ptr_limbs,
    u16_block_to_bytes,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub rs2_ptr: T,
    pub reads_aux: [MemoryReadAuxCols<T>; 2],
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

/// Reads instructions of the form OP a, b, c, d where \[a:8\]_d = \[b:8\]_d op \[c:8\]_d.
/// Operand d can only be 1, and there is no immediate support.
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(MultAdapterCols<u8>)]
pub struct MultAdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for MultAdapterAir {
    fn width(&self) -> usize {
        MultAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for MultAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        MinimalInstruction<AB::Expr>,
        2,
        1,
        REGISTER_NUM_LIMBS,
        REGISTER_NUM_LIMBS,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &MultAdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rs1_ptr),
                ),
                pack_u8_block::<AB>(&ctx.reads[0].clone()),
                timestamp_pp(),
                &local.reads_aux[0],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rs2_ptr),
                ),
                pack_u8_block::<AB>(&ctx.reads[1].clone()),
                timestamp_pp(),
                &local.reads_aux[1],
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rd_ptr),
                ),
                pack_u8_block::<AB>(&ctx.writes[0].clone()),
                timestamp_pp(),
                &local.writes_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    local.rd_ptr.into(),
                    local.rs1_ptr.into(),
                    local.rs2_ptr.into(),
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::ZERO,
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (openvm_instructions::program::DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> [AB::Var; 2] {
        let cols: &MultAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct MultAdapterFiller;

impl MultAdapterFiller {
    pub(crate) fn replay<F: PrimeField32, M>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut MultAdapterCols<F>,
        compute: impl FnOnce([[u8; REGISTER_NUM_LIMBS]; 2]) -> ReplayComputation<REGISTER_NUM_LIMBS, M>,
    ) -> Result<ReplayResult<REGISTER_NUM_LIMBS, M>, PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != REGISTER_AS || instruction.e.as_canonical_u32() != 0
        {
            return Err(PostflightError::new(
                "multiplication instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rs2_ptr = instruction.c.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rs2_u16_ptr = checked_register_u16_pointer(rs2_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(REGISTER_AS, rs1_u16_ptr)?;
        let rs2 = replay.read_u16(REGISTER_AS, rs2_u16_ptr)?;
        let inputs = [u16_block_to_bytes(rs1.value), u16_block_to_bytes(rs2.value)];
        let computation = compute(inputs);
        let output = computation.output;
        let write = replay.write_u16(REGISTER_AS, rd_u16_ptr, bytes_to_u16_block(output))?;
        replay.finish(from_pc.wrapping_add(DEFAULT_PC_STEP))?;

        adapter_row
            .writes_aux
            .set_prev_data(write.previous_value.map(F::from_u16));
        mem_helper.fill(
            write.previous_timestamp,
            write.timestamp,
            adapter_row.writes_aux.as_mut(),
        );
        mem_helper.fill(
            rs2.previous_timestamp,
            rs2.timestamp,
            adapter_row.reads_aux[1].as_mut(),
        );
        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux[0].as_mut(),
        );
        adapter_row.rs2_ptr = F::from_u32(rs2_ptr);
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = pc_to_limbs(from_pc).map(F::from_u32);

        Ok(ReplayResult {
            inputs,
            output,
            metadata: computation.metadata,
        })
    }
}
