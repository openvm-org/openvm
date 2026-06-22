use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::{ShiftOpcode, ShiftWOpcode};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    adapters::{RV64_WORD_U16_LIMBS, U16_BITS},
    shift_logical::run_shift_logical_u16,
    ShiftLogicalU16CoreRecord,
};

/// Executor for the RV64 `SLLW`/`SRLW` chip. The shift math is identical to the low 32 bits of the
/// register `SLL`/`SRL`, so trace generation reuses `run_shift_logical_u16` and
/// [`ShiftLogicalU16CoreRecord`]; the W adapter rebuilds the sign-extended 64-bit write. This is a
/// distinct type from `ShiftLogicalU16Executor` because its pure/metered interpreter (see
/// `execution.rs`) has W-specific 32->64 sign-extension semantics, which would conflict with
/// `ShiftLogicalU16Executor`'s blanket interpreter impls.
#[derive(Clone, Copy, derive_new::new)]
pub struct ShiftWLogicalExecutor<A> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA> PreflightExecutor<F, RA> for ShiftWLogicalExecutor<A>
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
            &'buf mut ShiftLogicalU16CoreRecord<RV64_WORD_U16_LIMBS, U16_BITS>,
        ),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!("{:?}", ShiftWOpcode::from_usize(opcode - self.offset))
    }

    fn execute(
        &self,
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;

        // `SLLW`/`SRLW` share opcode indices with `SLL`/`SRL`, so the core record and AIR offset
        // mapping reuse `ShiftOpcode` directly.
        let local_opcode = ShiftOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        debug_assert_ne!(local_opcode, ShiftOpcode::SRA);

        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);

        let [rs1, rs2] = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record)
            .into();

        let (output, _, _) =
            run_shift_logical_u16::<RV64_WORD_U16_LIMBS, U16_BITS>(local_opcode, &rs1, &rs2);

        core_record.b = rs1;
        core_record.c = rs2;
        core_record.local_opcode = local_opcode as u8;

        self.adapter.write(
            state.memory,
            instruction,
            [output].into(),
            &mut adapter_record,
        );

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);

        Ok(())
    }
}
