use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{BaseAluOpcode, BaseAluWOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    adapters::{RV64_WORD_U16_LIMBS, U16_BITS},
    add_sub::run_add_sub,
    AddSubCoreRecord,
};

/// Executor for the RV64 `ADDW`/`SUBW` chip. The arithmetic is identical to the low 32 bits of
/// `add_sub`, so trace generation reuses `run_add_sub` and [`AddSubCoreRecord`]; the adapter
/// rebuilds the sign-extended 64-bit write. This is a distinct type from `AddSubExecutor` because
/// its pure/metered interpreter (see `execution.rs`) has W-specific 32->64 sign-extension
/// semantics, which would conflict with `AddSubExecutor`'s blanket interpreter impls.
#[derive(Clone, Copy, derive_new::new)]
pub struct AddSubWExecutor<A> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA> PreflightExecutor<F, RA> for AddSubWExecutor<A>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData: Into<[[u16; RV64_WORD_U16_LIMBS]; 2]>,
            WriteData: From<[[u16; RV64_WORD_U16_LIMBS]; 1]>,
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (
            A::RecordMut<'buf>,
            &'buf mut AddSubCoreRecord<RV64_WORD_U16_LIMBS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", BaseAluWOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        // `ADDW`/`SUBW` share opcode indices with `ADD`/`SUB`, so the core record and AIR offset
        // mapping reuse `BaseAluOpcode` directly.
        let local_opcode = BaseAluOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert!(matches!(
            local_opcode,
            BaseAluOpcode::ADD | BaseAluOpcode::SUB
        ));
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        [core_record.b, core_record.c] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let rd = run_add_sub::<RV64_WORD_U16_LIMBS, U16_BITS>(
            local_opcode,
            &core_record.b,
            &core_record.c,
        );

        core_record.local_opcode = local_opcode as u8;

        self.adapter
            .write(state.memory, instruction, [rd].into(), &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
