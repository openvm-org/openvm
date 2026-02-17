use openvm_circuit::arch::{
    ExecuteFunc, ExecutionCtxTrait, InterpreterExecutor, InterpreterMeteredExecutor,
    MeteredExecutionCtxTrait, StaticProgramError,
};
use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;

use super::DeferralOutputExecutor;

impl<F: PrimeField32> InterpreterExecutor<F> for DeferralOutputExecutor {
    fn pre_compute_size(&self) -> usize {
        0
    }

    #[cfg(not(feature = "tco"))]
    fn pre_compute<Ctx>(
        &self,
        pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        Err(StaticProgramError::InvalidInstruction(pc))
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx>(
        &self,
        pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: ExecutionCtxTrait,
    {
        Err(StaticProgramError::InvalidInstruction(pc))
    }
}

impl<F: PrimeField32> InterpreterMeteredExecutor<F> for DeferralOutputExecutor {
    fn metered_pre_compute_size(&self) -> usize {
        0
    }

    #[cfg(not(feature = "tco"))]
    fn metered_pre_compute<Ctx>(
        &self,
        _air_idx: usize,
        pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        Err(StaticProgramError::InvalidInstruction(pc))
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx>(
        &self,
        _air_idx: usize,
        pc: u32,
        _inst: &Instruction<F>,
        _data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError>
    where
        Ctx: MeteredExecutionCtxTrait,
    {
        Err(StaticProgramError::InvalidInstruction(pc))
    }
}
