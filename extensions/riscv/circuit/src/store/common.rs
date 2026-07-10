use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::{
    rv64_bytes_to_u16_block, rv64_u16_block_to_bytes, STORE_WIDTH_BYTE, STORE_WIDTH_DOUBLEWORD,
    STORE_WIDTH_HALFWORD, STORE_WIDTH_WORD,
};

#[repr(C)]
#[derive(AlignedBytesBorrow, Clone, Copy, Debug)]
pub struct StoreRecord {
    pub read_data: [u16; BLOCK_FE_WIDTH],
    /// Previous contents of the two touched memory blocks; the second is all-zero unless the
    /// access crosses a block boundary.
    pub prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct StoreExecutor<A, const STORE_WIDTH: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const STORE_WIDTH: usize> PreflightExecutor<F, RA> for StoreExecutor<A, STORE_WIDTH>
where
    F: PrimeField32,
    A: 'static
        + AdapterTraceExecutor<
            F,
            ReadData = (([[u16; BLOCK_FE_WIDTH]; 2], [u16; BLOCK_FE_WIDTH]), u8),
            WriteData = [[u16; BLOCK_FE_WIDTH]; 2],
        >,
    for<'buf> RA: RecordArena<
        'buf,
        EmptyAdapterCoreLayout<F, A>,
        (A::RecordMut<'buf>, &'buf mut StoreRecord),
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
        state: VmStateMut<F, TracingMemory, RA>,
        instruction: &Instruction<F>,
    ) -> Result<(), ExecutionError> {
        let Instruction { opcode, .. } = instruction;
        let (mut adapter_record, core_record) = state.ctx.alloc(EmptyAdapterCoreLayout::new());

        A::start(*state.pc, state.memory, &mut adapter_record);
        let ((prev_data, read_data), shift_amount) =
            self.adapter
                .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        *core_record = StoreRecord {
            read_data,
            prev_data,
        };

        let write_data =
            store_write_data(local_opcode, read_data, prev_data, shift_amount as usize);
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

/// Returns the contents of the two written memory blocks for a store at any byte shift,
/// preserving previous bytes outside the store width. The second block is written back
/// unchanged unless the access crosses a block boundary.
pub(crate) fn store_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u16; BLOCK_FE_WIDTH],
    prev_data: [[u16; BLOCK_FE_WIDTH]; 2],
    byte_shift: usize,
) -> [[u16; BLOCK_FE_WIDTH]; 2] {
    debug_assert!(byte_shift < 2 * BLOCK_FE_WIDTH);
    let width = store_width_for_opcode(opcode);
    let mut bytes = [0u8; 4 * BLOCK_FE_WIDTH];
    bytes[..2 * BLOCK_FE_WIDTH].copy_from_slice(&rv64_u16_block_to_bytes(prev_data[0]));
    bytes[2 * BLOCK_FE_WIDTH..].copy_from_slice(&rv64_u16_block_to_bytes(prev_data[1]));
    let value = rv64_u16_block_to_bytes(read_data);
    bytes[byte_shift..byte_shift + width].copy_from_slice(&value[..width]);
    [
        rv64_bytes_to_u16_block(bytes[..2 * BLOCK_FE_WIDTH].try_into().unwrap()),
        rv64_bytes_to_u16_block(bytes[2 * BLOCK_FE_WIDTH..].try_into().unwrap()),
    ]
}

pub(crate) fn store_width_for_opcode(opcode: Rv64LoadStoreOpcode) -> usize {
    match opcode {
        STORED => STORE_WIDTH_DOUBLEWORD,
        STOREW => STORE_WIDTH_WORD,
        STOREH => STORE_WIDTH_HALFWORD,
        STOREB => STORE_WIDTH_BYTE,
        _ => unreachable!("unsupported store opcode: {opcode:?}"),
    }
}
