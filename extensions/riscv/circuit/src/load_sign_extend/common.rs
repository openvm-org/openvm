use openvm_circuit::arch::*;
use openvm_instructions::LocalOpcode;
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};

use crate::{
    adapters::{
        u16_cell_byte, LOAD_WIDTH_BYTE, LOAD_WIDTH_HALFWORD, LOAD_WIDTH_WORD, RV64_BYTE_SIGN_BIT,
        RV64_U16_SIGN_BIT,
    },
    load::LoadRecord,
};

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<A, const LOAD_WIDTH: usize> {
    adapter: A,
    pub offset: usize,
}

/// Returns the register write data for a signed load.
pub(crate) fn load_sign_extend_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u16; BLOCK_FE_WIDTH],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    let cell_shift = byte_shift / 2;
    match opcode {
        LOADW if byte_shift == 0 || byte_shift == 4 => {
            let sign = if read_data[cell_shift + 1] & RV64_U16_SIGN_BIT != 0 {
                u16::MAX
            } else {
                0
            };
            [read_data[cell_shift], read_data[cell_shift + 1], sign, sign]
        }
        LOADH if byte_shift == 0 || byte_shift == 2 || byte_shift == 4 || byte_shift == 6 => {
            let sign = if read_data[cell_shift] & RV64_U16_SIGN_BIT != 0 {
                u16::MAX
            } else {
                0
            };
            [read_data[cell_shift], sign, sign, sign]
        }
        LOADB if byte_shift < 8 => {
            let byte = u16_cell_byte(read_data[cell_shift], byte_shift % 2);
            if byte & RV64_BYTE_SIGN_BIT != 0 {
                [byte | 0xff00, u16::MAX, u16::MAX, u16::MAX]
            } else {
                [byte, 0, 0, 0]
            }
        }
        _ => unreachable!(
            "unaligned signed load not supported by this execution environment: {opcode:?}, byte_shift: {byte_shift}"
        ),
    }
}

pub(crate) fn load_sign_extend_width_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADW => LOAD_WIDTH_WORD,
        LOADH => LOAD_WIDTH_HALFWORD,
        LOADB => LOAD_WIDTH_BYTE,
        _ => unreachable!("unsupported signed load opcode: {opcode:?}"),
    }
}

impl<F, A, RA, const LOAD_WIDTH: usize> PreflightExecutor<F, RA>
    for LoadSignExtendExecutor<A, LOAD_WIDTH>
where
    F: openvm_stark_backend::p3_field::PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([u16; BLOCK_FE_WIDTH], [[u16; BLOCK_FE_WIDTH]; 2]), u8),
            WriteData = [u16; BLOCK_FE_WIDTH],
        >,
    for<'buf> RA:
        RecordArena<'buf, EmptyAdapterCoreLayout<F, A>, (A::RecordMut<'buf>, &'buf mut LoadRecord)>,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<openvm_circuit::system::memory::online::TracingMemory, RA>,
        instruction: &openvm_instructions::instruction::Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let openvm_instructions::instruction::Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);
        let ((_prev_data, read_data), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        *core_record = LoadRecord { read_data };

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        // Signed loads only support in-block shifts so far, so only the first block is used.
        let write_data =
            load_sign_extend_write_data(local_opcode, read_data[0], shift_amount as usize);
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state
            .pc
            .wrapping_add(openvm_instructions::program::DEFAULT_PC_STEP);
        Ok(())
    }
}
