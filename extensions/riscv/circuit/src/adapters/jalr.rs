use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState, Postflight,
        PostflightError, PostflightStep, SignedImmInstruction, VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryAuxColsFactory,
    },
};
use openvm_circuit_primitives::{utils::not, ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{program::DEFAULT_PC_STEP, riscv::RV64_REGISTER_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
};

use crate::adapters::{byte_ptr_to_u16_ptr, checked_register_u16_pointer};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64JalrAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rs1_ptr: T,
    pub rs1_aux_cols: MemoryReadAuxCols<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
    /// Only writes if `needs_write`.
    /// Sets `needs_write` to 0 iff `rd == x0`
    pub needs_write: T,
}

#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64JalrAdapterCols<u8>)]
pub struct Rv64JalrAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

impl<F: Field> BaseAir<F> for Rv64JalrAdapterAir {
    fn width(&self) -> usize {
        Rv64JalrAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64JalrAdapterAir {
    type Interface = BasicAdapterInterface<
        AB::Expr,
        SignedImmInstruction<AB::Expr>,
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
        let local_cols: &Rv64JalrAdapterCols<AB::Var> = local.borrow();

        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let mut timestamp_delta: usize = 0;
        let mut timestamp_pp = || {
            timestamp_delta += 1;
            timestamp + AB::Expr::from_usize(timestamp_delta - 1)
        };

        let write_count = local_cols.needs_write;

        builder.assert_bool(write_count);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(write_count);

        self.memory_bridge
            .read(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rs1_ptr),
                ),
                ctx.reads[0].clone(),
                timestamp_pp(),
                &local_cols.rs1_aux_cols,
            )
            .eval(builder, ctx.instruction.is_valid.clone());

        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rd_ptr),
                ),
                ctx.writes[0].clone(),
                timestamp_pp(),
                &local_cols.rd_aux_cols,
            )
            .eval(builder, write_count);

        let to_pc = ctx
            .to_pc
            .unwrap_or(local_cols.from_state.pc + AB::F::from_u32(DEFAULT_PC_STEP));

        // regardless of `needs_write`, must always execute instruction when `is_valid`.
        self.execution_bridge
            .execute(
                ctx.instruction.opcode,
                [
                    local_cols.rd_ptr.into(),
                    local_cols.rs1_ptr.into(),
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::ZERO,
                    write_count.into(),
                    ctx.instruction.imm_sign,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64JalrAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64JalrAdapterFiller;

impl Rv64JalrAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64JalrAdapterCols<F>,
        compute: impl FnOnce(
            u32,
            [u16; BLOCK_FE_WIDTH],
            u16,
            bool,
        ) -> Result<(u32, [u16; BLOCK_FE_WIDTH]), PostflightError>,
    ) -> Result<([u16; BLOCK_FE_WIDTH], u32, [u16; BLOCK_FE_WIDTH]), PostflightError> {
        let instruction = postflight.instruction(step);
        if instruction.d.as_canonical_u32() != RV64_REGISTER_AS || !instruction.e.is_zero() {
            return Err(PostflightError::new(
                "JALR instruction has invalid address spaces",
            ));
        }
        let needs_write = match instruction.f.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "JALR instruction has a non-boolean write enable",
                ));
            }
        };
        let imm_sign = match instruction.g.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "JALR instruction has a non-boolean immediate sign",
                ));
            }
        };
        let immediate = instruction.c.as_canonical_u32();
        let canonical_immediate = (immediate & 0x7ff) + u32::from(imm_sign) * 0xf800;
        if immediate != canonical_immediate {
            return Err(PostflightError::new(
                "JALR instruction has a non-canonical immediate",
            ));
        }

        let from_pc = postflight.pc(step);
        let from_timestamp = postflight.timestamp(step);
        let rs1_ptr = instruction.b.as_canonical_u32();
        let rd_ptr = instruction.a.as_canonical_u32();
        let rs1_u16_ptr = checked_register_u16_pointer(rs1_ptr)?;
        let rd_u16_ptr = checked_register_u16_pointer(rd_ptr)?;
        let mut replay = postflight.replay(step);
        let rs1 = replay.read_u16(RV64_REGISTER_AS, rs1_u16_ptr)?;
        let (to_pc, rd_data) = compute(from_pc, rs1.value, immediate as u16, imm_sign)?;

        adapter_row.needs_write = F::from_bool(needs_write);
        if needs_write {
            let write = replay.write_u16(RV64_REGISTER_AS, rd_u16_ptr, rd_data)?;
            adapter_row
                .rd_aux_cols
                .set_prev_data(write.previous_value.map(F::from_u16));
            mem_helper.fill(
                write.previous_timestamp,
                write.timestamp,
                adapter_row.rd_aux_cols.as_mut(),
            );
            adapter_row.rd_ptr = F::from_u32(rd_ptr);
        } else {
            replay.advance_timestamp(1)?;
            mem_helper.fill_zero(adapter_row.rd_aux_cols.as_mut());
            adapter_row.rd_ptr = F::ZERO;
        }
        replay.finish(to_pc & !1)?;

        mem_helper.fill(
            rs1.previous_timestamp,
            rs1.timestamp,
            adapter_row.rs1_aux_cols.as_mut(),
        );
        adapter_row.rs1_ptr = F::from_u32(rs1_ptr);
        adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
        adapter_row.from_state.pc = F::from_u32(from_pc);

        Ok((rs1.value, to_pc, rd_data))
    }
}
