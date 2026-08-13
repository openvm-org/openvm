use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction,
        Postflight, PostflightError, PostflightStep, VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::{pc_to_limbs, DEFAULT_PC_STEP},
    riscv::{IMM_AS, REGISTER_AS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::BaseAir,
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use super::{checked_register_u16_pointer, is_canonical_i12, reg_byte_ptr_to_cell_ptr_limbs};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct BaseAluImmU16AdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rs1_ptr: T,
    pub reads_aux: MemoryReadAuxCols<T>,
    pub writes_aux: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(BaseAluImmU16AdapterCols<u8>)]
pub struct BaseAluImmU16AdapterAir {
    pub(super) execution_bridge: ExecutionBridge,
    pub(super) memory_bridge: MemoryBridge,
}

impl<F: Field> BaseAir<F> for BaseAluImmU16AdapterAir {
    fn width(&self) -> usize {
        BaseAluImmU16AdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for BaseAluImmU16AdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        ImmInstruction<AB::Expr>,
        1,
        1,
        BLOCK_FE_WIDTH,
        BLOCK_FE_WIDTH,
    >;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local: &BaseAluImmU16AdapterCols<_> = local.borrow();
        let timestamp = local.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::F::from_usize(timestamp_delta - 1)
        };

        // rs1 register read (timestamp slot 0).
        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rs1_ptr),
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local.reads_aux,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        // rd write (timestamp slot 1).
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(REGISTER_AS),
                    reg_byte_ptr_to_cell_ptr_limbs::<AB>(local.rd_ptr),
                ),
                ctx.writes[0].clone(),
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
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(REGISTER_AS),
                    AB::Expr::from_u32(IMM_AS),
                ],
                local.from_state,
                AB::F::from_usize(timestamp_delta),
                (openvm_instructions::program::DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> [AB::Var; 2] {
        let cols: &BaseAluImmU16AdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct BaseAluImmU16AdapterFiller;

impl BaseAluImmU16AdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut BaseAluImmU16AdapterCols<F>,
        compute: impl FnOnce([u16; BLOCK_FE_WIDTH], u32) -> [u16; BLOCK_FE_WIDTH],
    ) -> Result<([u16; BLOCK_FE_WIDTH], [u16; BLOCK_FE_WIDTH]), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != REGISTER_AS
            || instruction.e.as_canonical_u32() != IMM_AS
        {
            return Err(PostflightError::new(
                "register-immediate ALU instruction has invalid address spaces",
            ));
        }
        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let immediate = instruction.c.as_canonical_u32();
        if !is_canonical_i12(immediate) {
            return Err(PostflightError::new(
                "register-immediate ALU instruction has a non-canonical immediate",
            ));
        }
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(REGISTER_AS, rs1_u16_ptr)?;
        let output = compute(rs1.value, immediate);
        let write = replay.write_u16(REGISTER_AS, rd_u16_ptr, output)?;
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
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.reads_aux.as_mut(),
        );
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.rd_ptr = F::from_u32(rd_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = pc_to_limbs(from_pc).map(F::from_u32);

        Ok((rs1.value, output))
    }
}
