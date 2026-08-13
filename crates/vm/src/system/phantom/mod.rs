//! Chip to handle phantom instructions.
//! The Air will always constrain a NOP which advances pc by DEFAULT_PC_STEP.
//! The runtime executor will execute different phantom instructions that may
//! affect trace generation based on the operand.
use std::{borrow::Borrow, sync::Arc};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{PhantomDiscriminant, VmOpcode};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use super::memory::online::GuestMemory;
use crate::arch::{ExecutionBridge, ExecutionState, PcIncOrSet, PhantomSubExecutor, Streams};

mod execution;
#[cfg(test)]
mod tests;
mod trace;

pub(crate) use trace::generate_trace_from_postflight;

/// PhantomAir still needs columns for each nonzero operand in a phantom instruction.
/// We currently allow `a,b,c` where the lower 16 bits of `c` are used as the [PhantomInstruction]
/// discriminant.
const NUM_PHANTOM_OPERANDS: usize = 3;

#[derive(Clone, Debug, ColumnsAir)]
#[columns_via(PhantomCols<u8>)]
pub struct PhantomAir {
    pub execution_bridge: ExecutionBridge,
    /// Global opcode for PhantomOpcode
    pub phantom_opcode: VmOpcode,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Copy, Clone, Serialize, Deserialize)]
pub struct PhantomCols<T> {
    pub pc: [T; 2],
    #[serde(with = "BigArray")]
    pub operands: [T; NUM_PHANTOM_OPERANDS],
    pub timestamp: T,
    pub is_valid: T,
}

impl<F: Field> BaseAir<F> for PhantomAir {
    fn width(&self) -> usize {
        PhantomCols::<F>::width()
    }
}
impl<F: Field> PartitionedBaseAir<F> for PhantomAir {}
impl<F: Field> BaseAirWithPublicValues<F> for PhantomAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for PhantomAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("window should have two elements");
        let &PhantomCols {
            pc,
            operands,
            timestamp,
            is_valid,
        } = (*local).borrow();

        builder.assert_bool(is_valid);
        self.execution_bridge
            .execute_and_increment_or_set_pc(
                self.phantom_opcode.to_field::<AB::F>(),
                operands,
                ExecutionState::<AB::Expr>::from_pc_limbs(pc, timestamp),
                AB::Expr::ONE,
                PcIncOrSet::Inc(AB::Expr::from_u32(
                    openvm_instructions::program::DEFAULT_PC_STEP,
                )),
            )
            .eval(builder, is_valid);
    }
}

/// Stateful executor that stores and dispatches all phantom sub-executors.
#[derive(Clone, derive_new::new)]
pub struct PhantomExecutor {
    pub(crate) phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor>>,
}

pub struct NopPhantomExecutor;
pub struct CycleStartPhantomExecutor;
pub struct CycleEndPhantomExecutor;

impl PhantomSubExecutor for NopPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams,
        _rng: &mut StdRng,
        _discriminant: PhantomDiscriminant,
        _a: u32,
        _b: u32,
        _c_upper: u16,
    ) -> eyre::Result<()> {
        Ok(())
    }
}

impl PhantomSubExecutor for CycleStartPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams,
        _rng: &mut StdRng,
        _discriminant: PhantomDiscriminant,
        _a: u32,
        _b: u32,
        _c_upper: u16,
    ) -> eyre::Result<()> {
        // Cycle tracking is implemented separately only in Preflight Execution
        Ok(())
    }
}

impl PhantomSubExecutor for CycleEndPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams,
        _rng: &mut StdRng,
        _discriminant: PhantomDiscriminant,
        _a: u32,
        _b: u32,
        _c_upper: u16,
    ) -> eyre::Result<()> {
        // Cycle tracking is implemented separately only in Preflight Execution
        Ok(())
    }
}
