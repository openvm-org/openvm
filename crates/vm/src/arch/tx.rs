//! Traits for AIR, execute, and trace transactions.
//! These are used with the integration API when a core implementation wishes
//! to be generic over the adapter. However, it is not _required_ to use these traits.

use openvm_stark_backend::p3_air::AirBuilder;
use serde::{de::DeserializeOwned, Serialize};

use crate::system::memory::{MemoryAddress, MemoryController};

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

pub trait AirTxRead<AB: AirBuilder, Data> {
    /// Return the memory address that was read from.
    /// The return type is needed to get the instruction.
    fn read(
        &mut self,
        builder: &mut AB,
        data: Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait AirTxMaybeRead<AB: AirBuilder, Data> {
    /// Apply constraints that may conditionally do a read.
    /// For example, either read from memory or handle immediates.
    fn maybe_read(
        &mut self,
        builder: &mut AB,
        data: Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait AirTxWrite<AB: AirBuilder, Data> {
    fn write(
        &mut self,
        builder: &mut AB,
        data: Data,
        multiplicity: impl Into<AB::Expr>,
    ) -> MemoryAddress<AB::Expr, AB::Expr>;
}

pub trait ExecuteTx {
    fn start(&mut self, from_pc: u32);

    /// Returns `to_pc`.
    fn end(&mut self) -> u32;
}

pub trait ExecuteTxRead<F, Data> {
    type Record: Clone + Send + Serialize + DeserializeOwned;

    fn read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> (Self::Record, Data);
}

pub trait ExecuteTxWrite<F, Data> {
    type Record: Clone + Send + Serialize + DeserializeOwned;

    /// Returns the previous data at the address.
    fn write(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
        data: Data,
    ) -> (Self::Record, Data);
}

pub trait ExecuteTxMaybeRead<F, Data> {
    type Record: Clone + Send + Serialize + DeserializeOwned;

    fn maybe_read(
        &mut self,
        memory: &mut MemoryController<F>,
        address: MemoryAddress<F, F>,
    ) -> (Self::Record, Data);
}

pub trait TraceTx<F> {
    fn start(&mut self);

    fn end(&mut self);
}

pub trait TraceTxRead<F> {
    type Record: Send + Serialize + DeserializeOwned;

    fn read(&mut self, record: Self::Record);
}

pub trait TraceTxMaybeRead<F> {
    type Record: Clone + Send + Serialize + DeserializeOwned;

    fn maybe_read(&mut self, record: Self::Record);
}

pub trait TraceTxWrite<F> {
    type Record: Clone + Send + Serialize + DeserializeOwned;

    fn write(&mut self, record: Self::Record);
}
