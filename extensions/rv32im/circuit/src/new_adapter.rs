use std::{
    borrow::{Borrow, BorrowMut},
    marker::PhantomData,
};

use openvm_circuit::{
    arch::{
        new_integration_api::{VmAdapter, VmAdapterAir},
        AirTx, AirTxMaybeRead, AirTxRead, AirTxWrite, ExecuteTx, ExecuteTxMaybeRead, ExecuteTxRead,
        ExecuteTxWrite, ExecutionState, SystemPort, TraceTx, TraceTxMaybeRead, TraceTxRead,
        TraceTxWrite, EXECUTION_STATE_WIDTH,
    },
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
    p3_air::{AirBuilder, BaseAir},
    p3_field::{FieldAlgebra, PrimeField32},
};

#[derive(Clone, Copy, derive_new::new)]
pub struct Rv32RegisterAdapter {
    port: SystemPort,

    // These are only needed to determine the adapter trace row width.
    // Keeping here to not have to make another struct.
    num_read: usize,
    num_read_or_imm: usize,
    num_write: usize,
}

impl<F> VmAdapter<F> for Rv32RegisterAdapter {
    type ExecuteTx = Rv32RegisterExecuteTx<F>;

    type TraceTx<'tx>
        = Rv32RegisterTraceTx<'tx, F>
    where
        Self: 'tx,
        F: 'tx;

    fn execute_tx(&self) -> Rv32RegisterExecuteTx<F> {
        Rv32RegisterExecuteTx {
            from_pc: None,
            phantom: PhantomData,
        }
    }

    fn trace_tx<'a>(
        &self,
        memory: &'a OfflineMemory<F>,
        row_adapter: &'a mut [F],
        from_state: ExecutionState<u32>,
    ) -> Rv32RegisterTraceTx<'a, F> {
        Rv32RegisterTraceTx {
            memory,
            row_buffer: row_adapter,
            pos: 0,
            from_state,
        }
    }
}

impl<F> BaseAir<F> for Rv32RegisterAdapter {
    fn width(&self) -> usize {
        STATE_WIDTH
            + self.num_read * READ_WIDTH
            + self.num_read_or_imm * READ_IMM_WIDTH
            + self.num_write * WRITE_WIDTH
    }
}

impl<AB: AirBuilder> VmAdapterAir<AB> for Rv32RegisterAdapter {
    type AirTx = Rv32RegisterAirTx<AB>;

    fn air_tx(&self, local_adapter: &[AB::Var]) -> Rv32RegisterAirTx<AB> {
        Rv32RegisterAirTx {
            port: self.port,
            row_buffer: local_adapter.to_vec(),
            pos: 0,
            cur_timestamp: None,
            instr_multiplicity: AB::Expr::ZERO,
            from_state: None,
        }
    }
}

pub struct Rv32RegisterAirTx<AB: AirBuilder> {
    port: SystemPort,
    // We use Vec instead of slice because there are some lifetime issues around
    // AB needing to outlive 'tx which Rust GATs can't handle yet.
    row_buffer: Vec<AB::Var>,
    pos: usize,
    pub cur_timestamp: Option<AB::Expr>,
    /// Multiplicity to use for program and execution bus
    instr_multiplicity: AB::Expr,
    from_state: Option<ExecutionState<AB::Expr>>,
    // Reminder: don't try to include `builder: &'a mut AB` because there
    // will be mutable borrow issues (you can't share a mutable reference)
}

impl<AB: AirBuilder> Drop for Rv32RegisterAirTx<AB> {
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

const STATE_WIDTH: usize = EXECUTION_STATE_WIDTH;
const READ_WIDTH: usize = size_of::<Rv32RegisterReadCols<u8>>();
const READ_IMM_WIDTH: usize = size_of::<Rv32RegOrImmReadCols<u8>>();
const WRITE_WIDTH: usize = size_of::<Rv32RegisterWriteCols<u8>>();

impl<AB: InteractionBuilder> AirTx<AB> for Rv32RegisterAirTx<AB> {
    fn start(&mut self, _builder: &mut AB, multiplicity: impl Into<AB::Expr>) {
        self.instr_multiplicity = multiplicity.into();
        let pos = self.pos;
        let from_state: &ExecutionState<AB::Var> = self.row_buffer[pos..pos + STATE_WIDTH].borrow();
        self.pos += STATE_WIDTH;
        self.cur_timestamp = Some(from_state.timestamp.into());
        self.from_state = Some(ExecutionState::new(from_state.pc, from_state.timestamp));
    }

    fn end_impl(
        &mut self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        to_pc: Option<AB::Expr>,
    ) {
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
        self.port.program_bus.send_instruction(
            builder,
            from_state.pc.clone(),
            opcode,
            operands,
            self.instr_multiplicity.clone(),
        );
        self.port.execution_bus.execute(
            builder,
            self.instr_multiplicity.clone(),
            from_state,
            to_state,
        );
    }
}

impl<AB: AirBuilder> Rv32RegisterAirTx<AB> {
    pub fn set_cur_timestamp(&mut self, timestamp: impl Into<AB::Expr>) {
        self.cur_timestamp = Some(timestamp.into());
    }
}

impl<AB: InteractionBuilder> AirTxRead<AB, [AB::Expr; RV32_REGISTER_NUM_LIMBS]>
    for Rv32RegisterAirTx<AB>
{
    fn read(
        &mut self,
        builder: &mut AB,
        data: [AB::Expr; RV32_REGISTER_NUM_LIMBS],
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr> {
        let pos = self.pos;
        let local: &Rv32RegisterReadCols<AB::Var> = self.row_buffer[pos..pos + READ_WIDTH].borrow();
        self.pos += READ_WIDTH;
        // Annoyance: we cannot make self.timestamp_pp() a function due to selective mutable borrow. This may be possible in a newer Rust version.
        let cur_timestamp = self.cur_timestamp.as_mut().unwrap();
        let timestamp = cur_timestamp.clone();
        *cur_timestamp = cur_timestamp.clone() + AB::Expr::ONE;
        let addr = MemoryAddress::new(
            AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
            local.ptr.into(),
        );
        self.port
            .memory_bridge
            .read(addr.clone(), data, timestamp, &local.aux)
            .eval(builder, multiplicity);
        addr
    }
}

impl<AB: InteractionBuilder> AirTxMaybeRead<AB, [AB::Expr; RV32_REGISTER_NUM_LIMBS]>
    for Rv32RegisterAirTx<AB>
{
    /// Memory bridge multiplicity is equal to `address_space`, which is 0 or 1.
    /// In particular, dummy rows should set `address_space` to 0
    /// (a row of all zeros will satisfy constraints).
    fn maybe_read(
        &mut self,
        builder: &mut AB,
        data: [AB::Expr; RV32_REGISTER_NUM_LIMBS],
        _multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr> {
        let pos = self.pos;
        let local: &Rv32RegOrImmReadCols<AB::Var> =
            self.row_buffer[pos..pos + READ_IMM_WIDTH].borrow();
        self.pos += READ_IMM_WIDTH;

        // if an immediate value, constrain that its 4-byte representation is correct
        let rs_sign = data[2].clone();
        let imm = data[0].clone()
            + data[1].clone() * AB::Expr::from_canonical_usize(1 << RV32_CELL_BITS)
            + rs_sign.clone() * AB::Expr::from_canonical_usize(1 << (2 * RV32_CELL_BITS));
        builder.assert_bool(local.address_space);
        let mut imm_when = builder.when(not(local.address_space));
        imm_when.assert_eq(local.ptr_or_imm, imm);
        imm_when.assert_eq(rs_sign.clone(), data[3].clone());
        imm_when.assert_zero(
            rs_sign.clone() * (AB::Expr::from_canonical_usize((1 << RV32_CELL_BITS) - 1) - rs_sign),
        );

        let cur_timestamp = self.cur_timestamp.as_mut().unwrap();
        let timestamp = cur_timestamp.clone();
        *cur_timestamp = cur_timestamp.clone() + AB::Expr::ONE;
        let addr = MemoryAddress::new(local.address_space.into(), local.ptr_or_imm.into());
        self.port
            .memory_bridge
            .read(addr.clone(), data, timestamp, &local.aux)
            .eval(builder, local.address_space);
        addr
    }
}

impl<AB: InteractionBuilder> AirTxWrite<AB, [AB::Expr; RV32_REGISTER_NUM_LIMBS]>
    for Rv32RegisterAirTx<AB>
{
    fn write(
        &mut self,
        builder: &mut AB,
        data: [AB::Expr; RV32_REGISTER_NUM_LIMBS],
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr> {
        let pos = self.pos;
        let local: &Rv32RegisterWriteCols<AB::Var> =
            self.row_buffer[pos..pos + WRITE_WIDTH].borrow();
        self.pos += WRITE_WIDTH;
        let cur_timestamp = self.cur_timestamp.as_mut().unwrap();
        let timestamp = cur_timestamp.clone();
        *cur_timestamp = cur_timestamp.clone() + AB::Expr::ONE;
        let addr = MemoryAddress::new(
            AB::Expr::from_canonical_u32(RV32_REGISTER_AS),
            local.ptr.into(),
        );
        self.port
            .memory_bridge
            .write(
                MemoryAddress::new(AB::F::from_canonical_u32(RV32_REGISTER_AS), local.ptr),
                data,
                timestamp,
                &local.aux,
            )
            .eval(builder, multiplicity);
        addr
    }
}

pub struct Rv32RegisterExecuteTx<F> {
    from_pc: Option<u32>,
    phantom: PhantomData<F>,
    // Reminder: don't try to include `memory: &'a mut MemoryController<F>`
    // because it will prevent shared borrowing of `MemoryController`
}

impl<F> Drop for Rv32RegisterExecuteTx<F> {
    fn drop(&mut self) {
        assert!(self.from_pc.is_none(), "Transaction was never ended");
    }
}

impl<F> ExecuteTx for Rv32RegisterExecuteTx<F> {
    fn start(&mut self, from_pc: u32) {
        self.from_pc = Some(from_pc);
    }

    fn end(&mut self) -> u32 {
        self.from_pc.take().unwrap() + DEFAULT_PC_STEP
    }
}

impl<F: PrimeField32> ExecuteTxRead<F, [F; RV32_REGISTER_NUM_LIMBS]> for Rv32RegisterExecuteTx<F> {
    type Record = RecordId;
    // Note[jpw]: we don't fix `address_space` because `F::from_canonical_u32` is not const. The instruction will already have `address_space` defined, so we pass it directly.
    fn read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> (RecordId, [F; RV32_REGISTER_NUM_LIMBS]) {
        debug_assert_eq!(address.address_space.as_canonical_u32(), RV32_REGISTER_AS);
        memory.read(address.address_space, address.pointer)
    }
}

impl<F: PrimeField32> ExecuteTxMaybeRead<F, [F; RV32_REGISTER_NUM_LIMBS]>
    for Rv32RegisterExecuteTx<F>
{
    /// The record has the form `(record_id, imm)` where
    /// - `record_id` is `None` if immediate.
    /// - `imm` is immediate field element if immediate, otherwise 0.
    type Record = (Option<RecordId>, F);
    /// Returns `Some(record_id)` if register or `None` if immediate.
    /// The `address.address_space` must be either 0 or 1.
    /// If 0, then `address.pointer` is interpretted as the immediate value.
    fn maybe_read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> ((Option<RecordId>, F), [F; RV32_REGISTER_NUM_LIMBS]) {
        let address_space = address.address_space;
        let ptr_or_imm = address.pointer;
        debug_assert!(
            address_space.as_canonical_u32() == RV32_IMM_AS
                || address_space.as_canonical_u32() == RV32_REGISTER_AS
        );
        if address_space.is_zero() {
            let imm_u32 = ptr_or_imm.as_canonical_u32();
            debug_assert_eq!(imm_u32 >> 24, 0);
            memory.increment_timestamp();
            (
                (None, ptr_or_imm),
                [
                    imm_u32 as u8,
                    (imm_u32 >> 8) as u8,
                    (imm_u32 >> 16) as u8,
                    (imm_u32 >> 16) as u8,
                ]
                .map(F::from_canonical_u8),
            )
        } else {
            let (id, data) = memory.read::<RV32_REGISTER_NUM_LIMBS>(address_space, ptr_or_imm);
            ((Some(id), F::ZERO), data)
        }
    }
}

impl<F: PrimeField32> ExecuteTxWrite<F, [F; RV32_REGISTER_NUM_LIMBS]> for Rv32RegisterExecuteTx<F> {
    type Record = RecordId;

    /// Returns `(id, prev_data)`
    fn write(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
        data: [F; RV32_REGISTER_NUM_LIMBS],
    ) -> (RecordId, [F; RV32_REGISTER_NUM_LIMBS]) {
        debug_assert_eq!(address.address_space.as_canonical_u32(), RV32_REGISTER_AS);
        memory.write(address.address_space, address.pointer, data)
    }
}

pub struct Rv32RegisterTraceTx<'a, F> {
    memory: &'a OfflineMemory<F>,
    from_state: ExecutionState<u32>,
    row_buffer: &'a mut [F],
    pos: usize,
}

impl<F: PrimeField32> TraceTx<F> for Rv32RegisterTraceTx<'_, F> {
    fn start(&mut self) {
        let pos = self.pos;
        let buffer: &mut ExecutionState<F> = self.row_buffer[pos..pos + STATE_WIDTH].borrow_mut();
        self.pos += STATE_WIDTH;
        buffer.pc = F::from_canonical_u32(self.from_state.pc);
        buffer.timestamp = F::from_canonical_u32(self.from_state.timestamp);
    }

    fn end(&mut self) {
        // do nothing
    }
}

impl<F: PrimeField32> TraceTxRead<F> for Rv32RegisterTraceTx<'_, F> {
    type Record = RecordId;

    fn read(&mut self, id: RecordId) {
        let pos = self.pos;
        let buffer: &mut Rv32RegisterReadCols<_> =
            self.row_buffer[pos..pos + READ_WIDTH].borrow_mut();
        self.pos += READ_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        let record = memory.record_by_id(id);
        buffer.ptr = record.pointer;
        aux_factory.generate_read_aux(record, &mut buffer.aux);
    }
}

impl<F: PrimeField32> TraceTxMaybeRead<F> for Rv32RegisterTraceTx<'_, F> {
    type Record = (Option<RecordId>, F);

    /// `id` is `None` if immediate.
    fn maybe_read(&mut self, (id, imm): (Option<RecordId>, F)) {
        let pos = self.pos;
        let buffer: &mut Rv32RegOrImmReadCols<_> =
            self.row_buffer[pos..pos + READ_IMM_WIDTH].borrow_mut();
        self.pos += READ_IMM_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        if let Some(record) = id.map(|id| memory.record_by_id(id)) {
            buffer.ptr_or_imm = record.pointer;
            buffer.address_space = record.address_space;
            aux_factory.generate_read_aux(record, &mut buffer.aux);
        } else {
            buffer.ptr_or_imm = imm;
            buffer.address_space = F::ZERO;
        }
    }
}

impl<F: PrimeField32> TraceTxWrite<F> for Rv32RegisterTraceTx<'_, F> {
    type Record = RecordId;

    fn write(&mut self, id: RecordId) {
        let pos = self.pos;
        let buffer: &mut Rv32RegisterWriteCols<_> =
            self.row_buffer[pos..pos + WRITE_WIDTH].borrow_mut();
        self.pos += WRITE_WIDTH;

        let memory = self.memory;
        let aux_factory = memory.aux_cols_factory();
        let record = memory.record_by_id(id);
        buffer.ptr = record.pointer;
        aux_factory.generate_write_aux(record, &mut buffer.aux);
    }
}
