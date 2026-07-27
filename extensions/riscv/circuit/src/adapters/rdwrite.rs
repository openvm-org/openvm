use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState, ImmInstruction,
        Postflight, PostflightError, PostflightStep, VmAdapterAir, BLOCK_FE_WIDTH,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
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

use crate::adapters::{byte_ptr_to_u16_ptr, checked_byte_ptr_to_u16_ptr_value};

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64RdWriteAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub rd_ptr: T,
    pub rd_aux_cols: MemoryWriteAuxCols<T, BLOCK_FE_WIDTH>,
}

#[repr(C)]
#[derive(Debug, Clone, AlignedBorrow, StructReflection)]
pub struct Rv64CondRdWriteAdapterCols<T> {
    pub inner: Rv64RdWriteAdapterCols<T>,
    pub needs_write: T,
}

/// This adapter doesn't read anything, and writes to \[a:8\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64RdWriteAdapterCols<u8>)]
pub struct Rv64RdWriteAdapterAir {
    pub(super) memory_bridge: MemoryBridge,
    pub(super) execution_bridge: ExecutionBridge,
}

/// This adapter doesn't read anything, and **maybe** writes to \[a:8\]_d, where d == 1
#[derive(Clone, Copy, Debug, derive_new::new, ColumnsAir)]
#[columns_via(Rv64CondRdWriteAdapterCols<u8>)]
pub struct Rv64CondRdWriteAdapterAir {
    inner: Rv64RdWriteAdapterAir,
}

impl<F: Field> BaseAir<F> for Rv64RdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv64RdWriteAdapterCols::<F>::width()
    }
}

impl<F: Field> BaseAir<F> for Rv64CondRdWriteAdapterAir {
    fn width(&self) -> usize {
        Rv64CondRdWriteAdapterCols::<F>::width()
    }
}

impl Rv64RdWriteAdapterAir {
    /// If `needs_write` is provided:
    /// - Only writes if `needs_write`.
    /// - Sets operand `f = needs_write` in the instruction.
    /// - Does not put any other constraints on `needs_write`
    ///
    /// Otherwise:
    /// - Writes if `ctx.instruction.is_valid`.
    /// - Sets operand `f` to default value of `0` in the instruction.
    #[allow(clippy::type_complexity)]
    fn conditional_eval<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        local_cols: &Rv64RdWriteAdapterCols<AB::Var>,
        ctx: AdapterAirContext<
            AB::Expr,
            BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, BLOCK_FE_WIDTH>,
        >,
        needs_write: Option<AB::Expr>,
    ) {
        let timestamp: AB::Var = local_cols.from_state.timestamp;
        let timestamp_delta = 1;
        let (write_count, f) = if let Some(needs_write) = needs_write {
            (needs_write.clone(), needs_write)
        } else {
            (ctx.instruction.is_valid.clone(), AB::Expr::ZERO)
        };
        self.memory_bridge
            .write(
                MemoryAddress::new(
                    AB::F::from_u32(RV64_REGISTER_AS),
                    byte_ptr_to_u16_ptr::<AB>(local_cols.rd_ptr),
                ),
                ctx.writes[0].clone(),
                timestamp,
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
                    AB::Expr::ZERO,
                    ctx.instruction.immediate,
                    AB::Expr::from_u32(RV64_REGISTER_AS),
                    AB::Expr::ZERO,
                    f,
                ],
                local_cols.from_state,
                ExecutionState {
                    pc: to_pc,
                    timestamp: timestamp + AB::F::from_usize(timestamp_delta),
                },
            )
            .eval(builder, ctx.instruction.is_valid);
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64RdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, BLOCK_FE_WIDTH>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv64RdWriteAdapterCols<AB::Var> = (*local).borrow();
        self.conditional_eval(builder, local_cols, ctx, None);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64RdWriteAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for Rv64CondRdWriteAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, ImmInstruction<AB::Expr>, 0, 1, 0, BLOCK_FE_WIDTH>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let local_cols: &Rv64CondRdWriteAdapterCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local_cols.needs_write);
        builder
            .when::<AB::Expr>(not(ctx.instruction.is_valid.clone()))
            .assert_zero(local_cols.needs_write);

        self.inner.conditional_eval(
            builder,
            &local_cols.inner,
            ctx,
            Some(local_cols.needs_write.into()),
        );
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &Rv64CondRdWriteAdapterCols<_> = local.borrow();
        cols.inner.from_state.pc
    }
}

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64RdWriteAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64RdWriteAdapterFiller;

/// This adapter doesn't read anything, and **maybe** writes to \[a:8\]_d, where d == 1
#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64CondRdWriteAdapterExecutor;

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv64CondRdWriteAdapterFiller;

impl Rv64RdWriteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64RdWriteAdapterCols<F>,
        compute: impl FnOnce(u32, u32) -> ([u16; BLOCK_FE_WIDTH], u32),
    ) -> Result<([u16; BLOCK_FE_WIDTH], u32), PostflightError> {
        if !postflight.instruction(step).f.is_zero() {
            return Err(PostflightError::new(
                "unconditional destination write has a nonzero enable flag",
            ));
        }
        replay_rd_write(postflight, step, mem_helper, adapter_row, true, compute)
    }
}

impl Rv64CondRdWriteAdapterFiller {
    pub(crate) fn replay<F: PrimeField32>(
        postflight: &Postflight<'_, F>,
        step: PostflightStep,
        mem_helper: &MemoryAuxColsFactory<F>,
        adapter_row: &mut Rv64CondRdWriteAdapterCols<F>,
        compute: impl FnOnce(u32, u32) -> ([u16; BLOCK_FE_WIDTH], u32),
    ) -> Result<([u16; BLOCK_FE_WIDTH], u32), PostflightError> {
        let needs_write = match postflight.instruction(step).f.as_canonical_u32() {
            0 => false,
            1 => true,
            _ => {
                return Err(PostflightError::new(
                    "conditional destination write has a non-boolean enable",
                ));
            }
        };
        adapter_row.needs_write = F::from_bool(needs_write);
        replay_rd_write(
            postflight,
            step,
            mem_helper,
            &mut adapter_row.inner,
            needs_write,
            compute,
        )
    }
}

fn replay_rd_write<F: PrimeField32>(
    postflight: &Postflight<'_, F>,
    step: PostflightStep,
    mem_helper: &MemoryAuxColsFactory<F>,
    adapter_row: &mut Rv64RdWriteAdapterCols<F>,
    needs_write: bool,
    compute: impl FnOnce(u32, u32) -> ([u16; BLOCK_FE_WIDTH], u32),
) -> Result<([u16; BLOCK_FE_WIDTH], u32), PostflightError> {
    let instruction = postflight.instruction(step);
    if instruction.d.as_canonical_u32() != RV64_REGISTER_AS || !instruction.e.is_zero() {
        return Err(PostflightError::new(
            "destination-write instruction has invalid address spaces",
        ));
    }
    let from_pc = postflight.pc(step);
    let from_timestamp = postflight.timestamp(step);
    let rd_ptr = instruction.a.as_canonical_u32();
    let immediate = instruction.c.as_canonical_u32();
    let (output, next_pc) = compute(from_pc, immediate);
    let rd_u16_ptr = checked_byte_ptr_to_u16_ptr_value(rd_ptr)?;
    let mut replay = postflight.replay(step);
    if needs_write {
        let write = replay.write_u16(RV64_REGISTER_AS, rd_u16_ptr, output)?;
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
    replay.finish(next_pc)?;
    adapter_row.from_state.timestamp = F::from_u32(from_timestamp);
    adapter_row.from_state.pc = F::from_u32(from_pc);

    Ok((output, next_pc))
}
