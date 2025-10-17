//! Chip to handle phantom instructions.
//! The Air will always constrain a NOP which advances pc by DEFAULT_PC_STEP.
//! The runtime executor will execute different phantom instructions that may
//! affect trace generation based on the operand.
use std::{
    borrow::{Borrow, BorrowMut},
    sync::Arc,
};

use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    instruction::Instruction, program::DEFAULT_PC_STEP, PhantomDiscriminant, SysPhantom,
    SystemOpcode, VmOpcode,
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, FieldAlgebra, PrimeField32},
    p3_matrix::Matrix,
    rap::{BaseAirWithPublicValues, PartitionedBaseAir},
};
use rand::rngs::StdRng;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;

use super::memory::online::{GuestMemory, TracingMemory};
use crate::{
    arch::{
        get_record_from_slice, EmptyMultiRowLayout, ExecutionBridge, ExecutionError,
        ExecutionState, PcIncOrSet, PhantomSubExecutor, PreflightExecutor, RecordArena, Streams,
        TraceFiller, VmChipWrapper, VmStateMut,
    },
    system::memory::MemoryAuxColsFactory,
};

mod execution;
#[cfg(test)]
mod tests;

/// PhantomAir still needs columns for each nonzero operand in a phantom instruction.
/// We currently allow `a,b,c` where the lower 16 bits of `c` are used as the [PhantomInstruction]
/// discriminant.
const NUM_PHANTOM_OPERANDS: usize = 3;

#[derive(Clone, Debug)]
pub struct PhantomAir {
    pub execution_bridge: ExecutionBridge,
    /// Global opcode for PhantomOpcode
    pub phantom_opcode: VmOpcode,
}

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Serialize, Deserialize)]
pub struct PhantomCols<T> {
    pub pc: T,
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
        let local = main.row_slice(0);
        let &PhantomCols {
            pc,
            operands,
            timestamp,
            is_valid,
        } = (*local).borrow();

        self.execution_bridge
            .execute_and_increment_or_set_pc(
                self.phantom_opcode.to_field::<AB::F>(),
                operands,
                ExecutionState::<AB::Expr>::new(pc, timestamp),
                AB::Expr::ONE,
                PcIncOrSet::Inc(AB::Expr::from_canonical_u32(DEFAULT_PC_STEP)),
            )
            .eval(builder, is_valid);
    }
}

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug, Clone)]
pub struct PhantomRecord {
    pub pc: u32,
    pub operands: [u32; NUM_PHANTOM_OPERANDS],
    pub timestamp: u32,
}

/// `PhantomChip` is a special executor because it is stateful and stores all the phantom
/// sub-executors.
#[derive(Clone, derive_new::new)]
pub struct PhantomExecutor<F> {
    pub(crate) phantom_executors: FxHashMap<PhantomDiscriminant, Arc<dyn PhantomSubExecutor<F>>>,
    phantom_opcode: VmOpcode,
}

pub struct PhantomFiller;
pub type PhantomChip<F> = VmChipWrapper<F, PhantomFiller>;

impl<F, RA> PreflightExecutor<F, RA> for PhantomExecutor<F>
where
    F: PrimeField32,
    for<'buf> RA: RecordArena<'buf, EmptyMultiRowLayout, &'buf mut PhantomRecord>,
{
    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let record: &mut PhantomRecord = state.ctx.alloc(EmptyMultiRowLayout::default());
        let pc = *state.pc;
        record.pc = pc;
        record.timestamp = state.memory.timestamp;
        let [a, b, c] = [instruction.a, instruction.b, instruction.c].map(|x| x.as_canonical_u32());
        record.operands = [a, b, c];

        debug_assert_eq!(instruction.opcode, self.phantom_opcode);
        let discriminant = PhantomDiscriminant(c as u16);
        if let Some(sys) = SysPhantom::from_repr(discriminant.0) {
            tracing::trace!("pc: {pc:#x} | system phantom: {sys:?}");
            match sys {
                SysPhantom::DebugPanic => {
                    #[cfg(all(
                        feature = "metrics",
                        any(debug_assertions, feature = "perf-metrics")
                    ))]
                    {
                        let metrics = state.metrics;
                        metrics.update_backtrace(pc);
                        if let Some(mut backtrace) = metrics.prev_backtrace.take() {
                            backtrace.resolve();
                            eprintln!("openvm program failure; backtrace:\n{:?}", backtrace);
                        } else {
                            eprintln!("openvm program failure; no backtrace");
                        }
                    }
                    return Err(ExecutionError::Fail {
                        pc,
                        msg: "DebugPanic",
                    });
                }
                #[cfg(feature = "perf-metrics")]
                SysPhantom::CtStart => {
                    let metrics = state.metrics;
                    if let Some(info) = metrics.debug_infos.get(pc) {
                        metrics.cycle_tracker.start(info.dsl_instruction.clone());
                    }
                }
                #[cfg(feature = "perf-metrics")]
                SysPhantom::CtEnd => {
                    let metrics = state.metrics;
                    if let Some(info) = metrics.debug_infos.get(pc) {
                        metrics.cycle_tracker.end(info.dsl_instruction.clone());
                    }
                }
                _ => {}
            }
        } else {
            let sub_executor = self.phantom_executors.get(&discriminant).unwrap();
            sub_executor
                .phantom_execute(
                    &state.memory.data,
                    state.streams,
                    state.rng,
                    discriminant,
                    a,
                    b,
                    (c >> 16) as u16,
                )
                .map_err(|err| ExecutionError::Phantom {
                    pc,
                    discriminant,
                    inner: err,
                })?;
        }
        *state.pc += DEFAULT_PC_STEP;
        state.memory.increment_timestamp();

        Ok(())
    }

    fn get_opcode_name(&self, _: usize) -> String {
        format!("{:?}", SystemOpcode::PHANTOM)
    }
}

impl<F: Field> TraceFiller<F> for PhantomFiller {
    fn fill_trace_row(&self, _mem_helper: &MemoryAuxColsFactory<F>, mut row_slice: &mut [F]) {
        // SAFETY: assume that row has size PhantomCols::<F>::width()
        let record: &PhantomRecord = unsafe { get_record_from_slice(&mut row_slice, ()) };
        let row: &mut PhantomCols<F> = row_slice.borrow_mut();
        // SAFETY: must assign in reverse order of column struct to prevent overwriting
        // borrowed data
        row.is_valid = F::ONE;
        row.timestamp = F::from_canonical_u32(record.timestamp);
        row.operands[2] = F::from_canonical_u32(record.operands[2]);
        row.operands[1] = F::from_canonical_u32(record.operands[1]);
        row.operands[0] = F::from_canonical_u32(record.operands[0]);
        row.pc = F::from_canonical_u32(record.pc)
    }
}

pub struct NopPhantomExecutor;
pub struct CycleStartPhantomExecutor;
pub struct CycleEndPhantomExecutor;

impl<F> PhantomSubExecutor<F> for NopPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams<F>,
        _rng: &mut StdRng,
        _discriminant: PhantomDiscriminant,
        _a: u32,
        _b: u32,
        _c_upper: u16,
    ) -> eyre::Result<()> {
        Ok(())
    }
}

impl<F> PhantomSubExecutor<F> for CycleStartPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams<F>,
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

impl<F> PhantomSubExecutor<F> for CycleEndPhantomExecutor {
    #[inline(always)]
    fn phantom_execute(
        &self,
        _memory: &GuestMemory,
        _streams: &mut Streams<F>,
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
