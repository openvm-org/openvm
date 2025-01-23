use openvm_stark_backend::p3_air::AirBuilder;

use crate::system::memory::{MemoryAddress, MemoryController, RecordId};

pub trait AirTx<AB: AirBuilder> {
    /// Start the constraints for a single instruction.
    /// The `end` or `end_jump` function should be called after all constraints
    /// for the instruction have been added.
    /// In conjunction with `end`, these function handles interactions
    /// on program and execution buses.
    fn start(&mut self, builder: &mut AB, multiplicity: impl Into<AB::Expr>);

    /// Ends transaction, incrementing the program counter by the default amount.
    /// The transaction **may** choose to modify `operands` before
    /// sending to the program bus.
    fn end(
        &mut self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
    ) {
        self.end_impl(builder, opcode, operands, None);
    }

    /// Ends transaction, jumping to the specified program counter.
    fn end_jump(
        &mut self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        to_pc: impl Into<AB::Expr>,
    ) {
        self.end_impl(builder, opcode, operands, Some(to_pc.into()));
    }

    /// Ends transaction, adding necessary interactions to program
    /// and execution bus. Timestamp is managed internally.
    ///
    /// In conjunction with `start`, these functions handle interactions
    /// on program and execution buses.
    fn end_impl(
        &mut self,
        builder: &mut AB,
        opcode: impl Into<AB::Expr>,
        operands: impl IntoIterator<Item = impl Into<AB::Expr>>,
        to_pc: Option<AB::Expr>,
    );
}

pub trait AirTxRead<AB: AirBuilder> {
    type Data;

    /// Return the memory address that was read from.
    /// The return type is needed to get the instruction.
    fn read(
        &mut self,
        builder: &mut AB,
        data: Self::Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait AirTxMaybeRead<AB: AirBuilder> {
    type Data;

    /// Apply constraints that may conditionally do a read.
    /// For example, either read from memory or handle immediates.
    fn maybe_read(
        &mut self,
        builder: &mut AB,
        data: Self::Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait AirTxWrite<AB: AirBuilder> {
    type Data;

    fn write(
        &mut self,
        builder: &mut AB,
        data: Self::Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait ExecuteTxRead<F> {
    type Data;

    fn read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> (RecordId, Self::Data);
}

pub trait ExecuteTxWrite<F> {
    type Data;

    /// Returns the previous data at the address.
    fn write(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
        data: Self::Data,
    ) -> (RecordId, Self::Data);
}

pub trait ExecuteTxMaybeRead<F> {
    type Data;

    fn maybe_read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> (Option<RecordId>, Self::Data);
}
