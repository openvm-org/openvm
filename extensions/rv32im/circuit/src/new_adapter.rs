use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{ExecutionState, SystemPort},
    system::memory::{
        offline_checker::{MemoryReadAuxCols, MemoryWriteAuxCols},
        MemoryAddress, MemoryController, OfflineMemory, RecordId,
    },
};
use openvm_circuit_primitives::utils::not;
use openvm_circuit_primitives_derive::AlignedBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV32_CELL_BITS, RV32_IMM_AS, RV32_REGISTER_AS, RV32_REGISTER_NUM_LIMBS},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::AirBuilder,
    p3_field::{FieldAlgebra, PrimeField32},
};

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32RegisterAdapter {
    port: SystemPort,
}

impl Rv32RegisterAdapter {
    pub fn execute_tx<'a, F>(
        &self,
        memory: &'a mut MemoryController<F>,
    ) -> Rv32RegisterExecuteTx<'a, F> {
        Rv32RegisterExecuteTx { memory }
    }

    pub fn air_tx<'a, AB: InteractionBuilder>(
        &self,
        builder: &'a mut AB,
        row_buffer: &'a [AB::Var],
    ) -> Rv32RegisterAirTx<'a, AB> {
        Rv32RegisterAirTx {
            port: self.port,
            builder,
            row_buffer,
            cur_timestamp: None,
            instr_multiplicity: AB::Expr::ZERO,
            from_state: None,
        }
    }
}

pub struct Rv32RegisterExecuteTx<'a, F> {
    memory: &'a mut MemoryController<F>,
}

impl<F: PrimeField32> Rv32RegisterExecuteTx<'_, F> {
    // Note[jpw]: we don't fix `address_space` because `F::from_canonical_u32` is not const. The instruction will already have `address_space` defined, so we pass it directly.
    pub fn read_register(
        &mut self,
        address_space: F,
        ptr: F,
    ) -> (RecordId, [F; RV32_REGISTER_NUM_LIMBS]) {
        debug_assert_eq!(address_space.as_canonical_u32(), RV32_REGISTER_AS);
        self.memory.read(address_space, ptr)
    }

    /// Returns `Some(record_id)` if register or `None` if immediate.
    pub fn read_register_or_imm(
        &mut self,
        address_space: F,
        ptr_or_imm: F,
    ) -> (Option<RecordId>, [F; RV32_REGISTER_NUM_LIMBS]) {
        debug_assert!(
            address_space.as_canonical_u32() == RV32_IMM_AS
                || address_space.as_canonical_u32() == RV32_REGISTER_AS
        );
        if address_space.is_zero() {
            let imm_u32 = ptr_or_imm.as_canonical_u32();
            debug_assert_eq!(imm_u32 >> 24, 0);
            self.memory.increment_timestamp();
            (
                None,
                [
                    imm_u32 as u8,
                    (imm_u32 >> 8) as u8,
                    (imm_u32 >> 16) as u8,
                    (imm_u32 >> 16) as u8,
                ]
                .map(F::from_canonical_u8),
            )
        } else {
            let (id, data) = self
                .memory
                .read::<RV32_REGISTER_NUM_LIMBS>(address_space, ptr_or_imm);
            (Some(id), data)
        }
    }

    /// Returns `(id, prev_data)`
    pub fn write_register(
        &mut self,
        address_space: F,
        ptr: F,
        data: [F; RV32_REGISTER_NUM_LIMBS],
    ) -> (RecordId, [F; RV32_REGISTER_NUM_LIMBS]) {
        debug_assert_eq!(address_space.as_canonical_u32(), RV32_REGISTER_AS);
        self.memory.write(address_space, ptr, data)
    }
}

pub struct Rv32RegisterAirTx<'a, AB: AirBuilder> {
    port: SystemPort,
    builder: &'a mut AB,
    row_buffer: &'a [AB::Var],
    pub cur_timestamp: Option<AB::Expr>,
    /// Multiplicity to use for program and execution bus
    instr_multiplicity: AB::Expr,
    from_state: Option<ExecutionState<AB::Expr>>,
}

impl<AB: AirBuilder> Drop for Rv32RegisterAirTx<'_, AB> {
    fn drop(&mut self) {
        assert!(self.cur_timestamp.is_none(), "Transaction was never ended");
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32RegisterReadCols<T> {
    pub ptr: T,
    pub aux: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32RegOrImmReadCols<T> {
    // Pointer if reading a register, immediate value otherwise
    pub ptr_or_imm: T,
    /// 1 if reading a register, 0 if an immediate (or dummy row)
    pub address_space: T,
    pub aux: MemoryReadAuxCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Rv32RegisterWriteCols<T> {
    pub ptr: T,
    pub aux: MemoryWriteAuxCols<T, RV32_REGISTER_NUM_LIMBS>,
}

const READ_WIDTH: usize = size_of::<Rv32RegisterReadCols<u8>>();
const READ_IMM_WIDTH: usize = size_of::<Rv32RegOrImmReadCols<u8>>();
const WRITE_WIDTH: usize = size_of::<Rv32RegisterWriteCols<u8>>();

impl<AB: InteractionBuilder> Rv32RegisterAirTx<'_, AB> {
    pub fn start(
        &mut self,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        from_state: ExecutionState<impl Into<AB::Expr>>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        self.instr_multiplicity = multiplicity.into();
        let from_state: ExecutionState<AB::Expr> =
            ExecutionState::new(from_state.pc, from_state.timestamp);
        self.port.program_bus.send_instruction(
            self.builder,
            from_state.pc.clone(),
            opcode,
            operands,
            self.instr_multiplicity.clone(),
        );
        self.cur_timestamp = Some(from_state.timestamp.clone());
        self.from_state = Some(from_state);
    }

    pub fn end(&mut self) {
        self.end_impl(None);
    }

    pub fn end_jump(&mut self, to_pc: impl Into<AB::Expr>) {
        self.end_impl(Some(to_pc.into()));
    }

    pub fn end_impl(&mut self, to_pc: Option<AB::Expr>) {
        let cur_timestamp = self
            .cur_timestamp
            .take()
            .expect("Transaction never started");
        let from_state = self.from_state.take().unwrap();
        let to_pc =
            to_pc.unwrap_or(from_state.pc.clone() + AB::Expr::from_canonical_u32(DEFAULT_PC_STEP));
        let to_state = ExecutionState {
            pc: to_pc,
            timestamp: cur_timestamp,
        };
        self.port.execution_bus.execute(
            self.builder,
            self.instr_multiplicity.clone(),
            from_state,
            to_state,
        );
    }

    pub fn set_cur_timestamp(&mut self, timestamp: impl Into<AB::Expr>) {
        self.cur_timestamp = Some(timestamp.into());
    }

    fn timestamp_pp(&mut self) -> AB::Expr {
        let cur_timestamp = self.cur_timestamp.as_mut().unwrap();
        let t = cur_timestamp.clone();
        *cur_timestamp = cur_timestamp.clone() + AB::Expr::ONE;
        t
    }

    pub fn read_register(
        &mut self,
        data: [AB::Expr; RV32_REGISTER_NUM_LIMBS],
        multiplicity: impl Into<AB::Expr>,
    ) {
        let (local, remaining) = self.row_buffer.split_at(READ_WIDTH);
        self.row_buffer = remaining;
        let local: &Rv32RegisterReadCols<AB::Var> = local.borrow();
        let timestamp = self.timestamp_pp();
        self.port
            .memory_bridge
            .read(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.ptr),
                data,
                timestamp,
                &local.aux,
            )
            .eval(self.builder, multiplicity);
    }

    /// Memory bridge multiplicity is equal to `address_space`, which is 0 or 1.
    /// In particular, dummy rows should set `address_space` to 0
    /// (a row of all zeros will satisfy constraints).
    pub fn read_register_or_imm(&mut self, data: [AB::Expr; RV32_REGISTER_NUM_LIMBS]) {
        let (local, remaining) = self.row_buffer.split_at(READ_IMM_WIDTH);
        self.row_buffer = remaining;
        let local: &Rv32RegOrImmReadCols<AB::Var> = local.borrow();

        // if an immediate value, constrain that its 4-byte representation is correct
        let rs_sign = data[2].clone();
        let imm = data[0].clone()
            + data[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        self.builder.assert_bool(local.address_space);
        let mut imm_when = self.builder.when(not(local.address_space));
        imm_when.assert_eq(local.ptr_or_imm, imm);
        imm_when.assert_eq(rs_sign.clone(), data[3].clone());
        imm_when.assert_zero(
            rs_sign.clone() * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs_sign),
        );

        let timestamp = self.timestamp_pp();
        self.port
            .memory_bridge
            .read(
                MemoryAddress::new(local.address_space, local.ptr_or_imm),
                data,
                timestamp,
                &local.aux,
            )
            .eval(self.builder, local.address_space);
    }

    pub fn write_register(
        &mut self,
        data: [AB::Expr; RV32_REGISTER_NUM_LIMBS],
        multiplicity: impl Into<AB::Expr>,
    ) {
        let (local, remaining) = self.row_buffer.split_at(WRITE_WIDTH);
        self.row_buffer = remaining;
        let local: &Rv32RegisterWriteCols<AB::Var> = local.borrow();
        let timestamp = self.timestamp_pp();
        self.port
            .memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.ptr),
                data,
                timestamp,
                &local.aux,
            )
            .eval(self.builder, multiplicity);
    }
}

pub struct Rv32RegisterTraceTx<'a, F> {
    memory: &'a OfflineMemory<F>,
    row_buffer: &'a mut [F],
    pos: usize,
}

impl<F: PrimeField32> Rv32RegisterTraceTx<'_, F> {
    pub fn read_register(&mut self, id: RecordId) {
        let pos = self.pos;
        let row_slice: &mut Rv32RegisterReadCols<_> =
            self.row_buffer[pos..pos + READ_WIDTH].borrow_mut();
        self.pos += READ_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        let record = memory.record_by_id(id);
        row_slice.ptr = record.pointer;
        aux_factory.generate_read_aux(record, &mut row_slice.aux);
    }

    /// `id` is `None` if immediate.
    pub fn read_register_or_imm(&mut self, id: Option<RecordId>, imm: F) {
        let pos = self.pos;
        let row_slice: &mut Rv32RegOrImmReadCols<_> =
            self.row_buffer[pos..pos + READ_IMM_WIDTH].borrow_mut();
        self.pos += READ_IMM_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        if let Some(record) = id.map(|id| memory.record_by_id(id)) {
            row_slice.ptr_or_imm = record.pointer;
            row_slice.address_space = record.address_space;
            aux_factory.generate_read_aux(record, &mut row_slice.aux);
        } else {
            row_slice.ptr_or_imm = imm;
            row_slice.address_space = F::ZERO;
        }
    }

    pub fn write_register(&mut self, id: RecordId) {
        let pos = self.pos;
        let row_slice: &mut Rv32RegisterWriteCols<_> =
            self.row_buffer[pos..pos + WRITE_WIDTH].borrow_mut();
        self.pos += WRITE_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        let record = memory.record_by_id(id);
        row_slice.ptr = record.pointer;
        aux_factory.generate_write_aux(record, &mut row_slice.aux);
    }
}
