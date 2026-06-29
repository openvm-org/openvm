use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::u16_cell_byte;
pub(crate) use crate::adapters::{
    RV64_ACCESS_SIZE_BYTE as KIND_BYTE, RV64_ACCESS_SIZE_DOUBLEWORD as KIND_DOUBLEWORD,
    RV64_ACCESS_SIZE_HALFWORD as KIND_HALFWORD, RV64_ACCESS_SIZE_WORD as KIND_WORD,
};

#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy, Debug)]
pub struct LoadRecord {
    pub local_opcode: u8,
    pub shift_amount: u8,
    pub read_data: [u16; BLOCK_FE_WIDTH],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadExecutor<A, const KIND: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const KIND: usize> PreflightExecutor<F, RA> for LoadExecutor<A, KIND>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([u16; BLOCK_FE_WIDTH], [u16; BLOCK_FE_WIDTH]), u8),
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
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);
        let ((_prev_data, read_data), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        *core_record = LoadRecord {
            local_opcode: local_opcode as u8,
            shift_amount,
            read_data,
        };

        let write_data = load_write_data(local_opcode, read_data, shift_amount as usize);
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

/// Returns the register write data for an unsigned load.
pub(crate) fn load_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u16; BLOCK_FE_WIDTH],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    let cell_shift = byte_shift / 2;
    match opcode {
        LOADD if byte_shift == 0 => read_data,
        LOADWU if byte_shift == 0 || byte_shift == 4 => [
            read_data[cell_shift],
            read_data[cell_shift + 1],
            0,
            0,
        ],
        LOADHU if byte_shift == 0 || byte_shift == 2 || byte_shift == 4 || byte_shift == 6 => {
            [read_data[cell_shift], 0, 0, 0]
        }
        LOADBU if byte_shift < 8 => {
            let byte = u16_cell_byte(read_data[cell_shift], byte_shift % 2);
            [byte, 0, 0, 0]
        }
        _ => unreachable!(
            "unaligned load not supported by this execution environment: {opcode:?}, byte_shift: {byte_shift}"
        ),
    }
}

pub(crate) fn load_kind_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        LOADD => KIND_DOUBLEWORD,
        LOADWU => KIND_WORD,
        LOADHU => KIND_HALFWORD,
        LOADBU => KIND_BYTE,
        _ => unreachable!("unsupported unsigned load opcode: {opcode:?}"),
    }
}
