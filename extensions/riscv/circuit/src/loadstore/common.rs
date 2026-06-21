use openvm_circuit::{arch::*, system::memory::online::TracingMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_riscv_transpiler::Rv64LoadStoreOpcode::{self, *};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::adapters::LoadStoreInstruction;

pub const KIND_BYTE: usize = 0;
pub const KIND_HALFWORD: usize = 1;
pub const KIND_WORD: usize = 2;
pub const KIND_DOUBLEWORD: usize = 3;

pub(crate) const BYTE_BITS: usize = 8;
pub(crate) const BYTE_MASK: u16 = 0xff;
pub(crate) const SIGN_BYTE: u16 = 1 << 7;
pub(crate) const SIGN_U16: u16 = 1 << 15;

#[repr(C)]
#[derive(AlignedBytesBorrow, Debug)]
pub struct LoadStoreRecord {
    pub local_opcode: u8,
    pub shift_amount: u8,
    pub read_data: [u16; BLOCK_FE_WIDTH],
    pub prev_data: [u16; BLOCK_FE_WIDTH],
}

#[derive(Clone, Copy, derive_new::new)]
pub struct LoadStoreExecutor<A, const KIND: usize> {
    adapter: A,
    pub offset: usize,
}

impl<F, A, RA, const KIND: usize> PreflightExecutor<F, RA> for LoadStoreExecutor<A, KIND>
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
        (A::RecordMut<'buf>, &'buf mut LoadStoreRecord),
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
        (
            (core_record.prev_data, core_record.read_data),
            core_record.shift_amount,
        ) = self
            .adapter
            .read(state.memory, instruction, &mut adapter_record);

        let local_opcode = Rv64LoadStoreOpcode::from_usize(opcode.local_opcode_idx(self.offset));
        core_record.local_opcode = local_opcode as u8;

        let write_data = run_write_data(
            local_opcode,
            core_record.read_data,
            core_record.prev_data,
            core_record.shift_amount as usize,
        );
        self.adapter
            .write(state.memory, instruction, write_data, &mut adapter_record);

        *state.pc = state.pc.wrapping_add(DEFAULT_PC_STEP);
        Ok(())
    }
}

pub(crate) fn byte_from_cell(cell: u16, byte_idx: usize) -> u16 {
    (cell >> (BYTE_BITS * byte_idx)) & BYTE_MASK
}

pub(crate) fn replace_byte(cell: u16, byte_idx: usize, byte: u16) -> u16 {
    debug_assert!(byte <= BYTE_MASK);
    let shift = BYTE_BITS * byte_idx;
    (cell & !(BYTE_MASK << shift)) | (byte << shift)
}

pub(crate) fn run_write_data(
    opcode: Rv64LoadStoreOpcode,
    read_data: [u16; BLOCK_FE_WIDTH],
    prev_data: [u16; BLOCK_FE_WIDTH],
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
        LOADW if byte_shift == 0 || byte_shift == 4 => {
            let sign = if read_data[cell_shift + 1] & SIGN_U16 != 0 {
                u16::MAX
            } else {
                0
            };
            [read_data[cell_shift], read_data[cell_shift + 1], sign, sign]
        }
        LOADHU if byte_shift == 0 || byte_shift == 2 || byte_shift == 4 || byte_shift == 6 => {
            [read_data[cell_shift], 0, 0, 0]
        }
        LOADH if byte_shift == 0 || byte_shift == 2 || byte_shift == 4 || byte_shift == 6 => {
            let sign = if read_data[cell_shift] & SIGN_U16 != 0 {
                u16::MAX
            } else {
                0
            };
            [read_data[cell_shift], sign, sign, sign]
        }
        LOADBU if byte_shift < 8 => {
            let byte = byte_from_cell(read_data[cell_shift], byte_shift % 2);
            [byte, 0, 0, 0]
        }
        LOADB if byte_shift < 8 => {
            let byte = byte_from_cell(read_data[cell_shift], byte_shift % 2);
            if byte & SIGN_BYTE != 0 {
                [byte | 0xff00, u16::MAX, u16::MAX, u16::MAX]
            } else {
                [byte, 0, 0, 0]
            }
        }
        STORED if byte_shift == 0 => read_data,
        STOREW if byte_shift == 0 || byte_shift == 4 => {
            let mut write_data = prev_data;
            write_data[cell_shift] = read_data[0];
            write_data[cell_shift + 1] = read_data[1];
            write_data
        }
        STOREH if byte_shift == 0 || byte_shift == 2 || byte_shift == 4 || byte_shift == 6 => {
            let mut write_data = prev_data;
            write_data[cell_shift] = read_data[0];
            write_data
        }
        STOREB if byte_shift < 8 => {
            let mut write_data = prev_data;
            let byte = byte_from_cell(read_data[0], 0);
            write_data[cell_shift] = replace_byte(prev_data[cell_shift], byte_shift % 2, byte);
            write_data
        }
        _ => unreachable!(
            "unaligned memory access not supported by this execution environment: {opcode:?}, byte_shift: {byte_shift}"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adapter_context<AB, I>(
    is_valid: AB::Expr,
    is_load: AB::Expr,
    expected_opcode: AB::Expr,
    load_shift_amount: AB::Expr,
    store_shift_amount: AB::Expr,
    read_data: [AB::Var; BLOCK_FE_WIDTH],
    prev_data: [AB::Var; BLOCK_FE_WIDTH],
    write_data: [AB::Expr; BLOCK_FE_WIDTH],
) -> AdapterAirContext<AB::Expr, I>
where
    AB: openvm_stark_backend::interaction::InteractionBuilder,
    I: VmAdapterInterface<AB::Expr>,
    I::Reads: From<([AB::Var; BLOCK_FE_WIDTH], [AB::Expr; BLOCK_FE_WIDTH])>,
    I::Writes: From<[[AB::Expr; BLOCK_FE_WIDTH]; 1]>,
    I::ProcessedInstruction: From<LoadStoreInstruction<AB::Expr>>,
{
    AdapterAirContext {
        to_pc: None,
        reads: (prev_data, read_data.map(|x| x.into())).into(),
        writes: [write_data].into(),
        instruction: LoadStoreInstruction {
            is_valid,
            opcode: expected_opcode,
            is_load,
            load_shift_amount,
            store_shift_amount,
        }
        .into(),
    }
}
