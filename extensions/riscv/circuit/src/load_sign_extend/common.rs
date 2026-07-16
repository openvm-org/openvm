use openvm_circuit::{
    arch::{
        AdapterTraceExecutor, EmptyAdapterCoreLayout, ExecutionError, PreflightExecutor,
        RecordArena, VmStateMut, BLOCK_FE_WIDTH,
    },
    system::memory::online::TracingMemory,
};
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, LOADB, LOADH, LOADW};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    adapters::{
        rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, LOAD_WIDTH_BYTE, LOAD_WIDTH_HALFWORD,
        LOAD_WIDTH_WORD,
    },
    load::{LoadByteRecord, LoadRecord},
};

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadSignExtendExecutor<A, const LOAD_WIDTH: usize, const NUM_BLOCKS: usize = 2> {
    adapter: A,
    pub offset: usize,
}

/// Returns the register write data for a signed load at any byte shift, including accesses that
/// span both blocks.
pub(crate) fn load_sign_extend_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [u16; BLOCK_FE_WIDTH] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let width = load_sign_extend_width_for_opcode(opcode);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&rv64_u16_block_to_bytes(read_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&rv64_u16_block_to_bytes(read_data[1]));
    let sign = (bytes[byte_shift + width - 1] as i8) < 0;
    let mut loaded = [if sign { 0xff } else { 0 }; 2 * BLOCK_FE_WIDTH];
    loaded[..width].copy_from_slice(&bytes[byte_shift..byte_shift + width]);
    rv64_bytes_to_u16_block(loaded)
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
    F: PrimeField32,
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
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);
        let ((_prev_data, read_data), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        *core_record = LoadRecord { read_data };

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        let write_data =
            load_sign_extend_write_data(local_opcode, read_data, shift_amount as usize);
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

impl<F, A, RA> PreflightExecutor<F, RA> for LoadSignExtendExecutor<A, LOAD_WIDTH_BYTE, 1>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([u16; BLOCK_FE_WIDTH], [u16; BLOCK_FE_WIDTH]), u8),
            WriteData = [u16; BLOCK_FE_WIDTH],
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut LoadByteRecord),
    >,
{
    fn get_opcode_name(&self, opcode: usize) -> String {
        format!(
            "{:?}",
            Rv64LoadStoreOpcode::from_usize(opcode - self.offset)
        )
    }

    fn execute(
        &self,
        state: VmStateMut<TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);
        let ((_prev_data, read_data), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        *core_record = LoadByteRecord { read_data };
        let write_data = load_sign_extend_write_data(
            LOADB,
            [read_data, [0; BLOCK_FE_WIDTH]],
            shift_amount as usize,
        );
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}
