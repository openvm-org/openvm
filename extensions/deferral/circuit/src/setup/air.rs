use std::borrow::Borrow;

use openvm_circuit::{
    arch::{
        AdapterAirContext, BasicAdapterInterface, ExecutionBridge, ExecutionState,
        MinimalInstruction, VmAdapterAir, VmAdapterInterface, VmCoreAir,
    },
    system::memory::{
        offline_checker::{MemoryBridge, MemoryWriteAuxCols},
        MemoryAddress,
    },
};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_deferral_transpiler::DeferralOpcode;
use openvm_instructions::{program::DEFAULT_PC_STEP, LocalOpcode, NATIVE_AS};
use openvm_stark_backend::{
    interaction::InteractionBuilder, p3_air::BaseAir, p3_field::PrimeCharacteristicRing,
    BaseAirWithPublicValues,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::utils::{split_memory_ops, DIGEST_MEMORY_OPS, MEMORY_OP_SIZE};

// ========================= CORE ==============================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralSetupCoreCols<T> {
    pub is_valid: T,
}

#[derive(Copy, Clone, Debug, derive_new::new)]
pub struct DeferralSetupCoreAir<F> {
    expected_def_vks_commit: [F; DIGEST_SIZE],
}

impl<F: Sync> BaseAir<F> for DeferralSetupCoreAir<F> {
    fn width(&self) -> usize {
        DeferralSetupCoreCols::<F>::width()
    }
}
impl<F: Sync> BaseAirWithPublicValues<F> for DeferralSetupCoreAir<F> {}

impl<AB, I> VmCoreAir<AB, I> for DeferralSetupCoreAir<AB::F>
where
    AB: InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<[[AB::Expr; 0]; 0]>,
    I::Writes: From<[[AB::Expr; DIGEST_SIZE]; 1]>,
    I::ProcessedInstruction: From<MinimalInstruction<AB::Expr>>,
{
    fn eval(
        &self,
        builder: &mut AB,
        local_core: &[AB::Var],
        _from_pc: AB::Var,
    ) -> AdapterAirContext<AB::Expr, I> {
        let cols: &DeferralSetupCoreCols<_> = local_core.borrow();
        builder.assert_bool(cols.is_valid);

        AdapterAirContext {
            to_pc: None,
            reads: [].into(),
            writes: [self.expected_def_vks_commit.map(Into::into)].into(),
            instruction: MinimalInstruction {
                is_valid: cols.is_valid.into(),
                opcode: AB::Expr::from_usize(DeferralOpcode::SETUP.global_opcode_usize()),
            }
            .into(),
        }
    }

    fn start_offset(&self) -> usize {
        DeferralOpcode::CLASS_OFFSET
    }
}

// ========================= ADAPTER ==============================

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct DeferralSetupAdapterCols<T> {
    pub from_state: ExecutionState<T>,
    pub write_aux: [MemoryWriteAuxCols<T, MEMORY_OP_SIZE>; DIGEST_MEMORY_OPS],
}

#[derive(Clone, Copy, Debug, derive_new::new)]
pub struct DeferralSetupAdapterAir {
    pub execution_bridge: ExecutionBridge,
    pub memory_bridge: MemoryBridge,
}

impl<F> BaseAir<F> for DeferralSetupAdapterAir {
    fn width(&self) -> usize {
        DeferralSetupAdapterCols::<F>::width()
    }
}

impl<AB: InteractionBuilder> VmAdapterAir<AB> for DeferralSetupAdapterAir {
    type Interface =
        BasicAdapterInterface<AB::Expr, MinimalInstruction<AB::Expr>, 0, 1, 0, DIGEST_SIZE>;

    fn eval(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        ctx: AdapterAirContext<AB::Expr, Self::Interface>,
    ) {
        let cols: &DeferralSetupAdapterCols<_> = local.borrow();

        // The SETUP opcode should always write to the beginning of the native
        // address space.
        let address_space = AB::Expr::from_u32(NATIVE_AS);
        let pointer = AB::Expr::ZERO;
        let [write_data] = ctx.writes;
        let write_chunks = split_memory_ops::<_, DIGEST_SIZE, DIGEST_MEMORY_OPS>(write_data);

        for (chunk_idx, (data, aux)) in write_chunks.into_iter().zip(&cols.write_aux).enumerate() {
            self.memory_bridge
                .write(
                    MemoryAddress::new(
                        address_space.clone(),
                        pointer.clone() + AB::Expr::from_usize(chunk_idx * MEMORY_OP_SIZE),
                    ),
                    data,
                    cols.from_state.timestamp + AB::Expr::from_usize(chunk_idx),
                    aux,
                )
                .eval(builder, ctx.instruction.is_valid.clone());
        }

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                ctx.instruction.opcode,
                [
                    pointer,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    address_space,
                    AB::Expr::ZERO,
                ],
                cols.from_state,
                AB::Expr::from_usize(DIGEST_MEMORY_OPS),
                (DEFAULT_PC_STEP, ctx.to_pc),
            )
            .eval(builder, ctx.instruction.is_valid);
    }

    fn get_from_pc(&self, local: &[AB::Var]) -> AB::Var {
        let cols: &DeferralSetupAdapterCols<_> = local.borrow();
        cols.from_state.pc
    }
}
