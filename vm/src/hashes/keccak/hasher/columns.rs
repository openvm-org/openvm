use core::mem::size_of;
use std::{
    array::from_fn,
    borrow::{Borrow, BorrowMut},
};

use afs_derive::AlignedBorrow;
use p3_air::AirBuilder;
use p3_keccak_air::{KeccakCols as KeccakPermCols, NUM_KECCAK_COLS as NUM_KECCAK_PERM_COLS};

use super::{
    KECCAK_ABSORB_READS, KECCAK_DIGEST_WRITES, KECCAK_EXECUTION_READS, KECCAK_RATE_BYTES,
    KECCAK_RATE_U16S,
};
use crate::memory::offline_checker::{
    bridge::MemoryOfflineChecker,
    columns::{MemoryReadAuxCols, MemoryWriteAuxCols},
};

#[derive(Clone, Copy, Debug)]
pub struct KeccakVmColsRef<'a, T> {
    /// Columns for keccak-f permutation
    pub inner: &'a KeccakPermCols<T>,
    /// Columns for sponge and padding
    pub sponge: &'a KeccakSpongeCols<T>,
    /// Columns for opcode interface and operand memory access
    pub opcode: &'a KeccakOpcodeCols<T>,
    /// Auxiliary columns for offline memory checking
    /// This should be convertable to [KeccakMemoryCols]
    pub mem_oc: &'a [T],
}

#[derive(Debug)]
pub struct KeccakVmColsMut<'a, T> {
    /// Columns for keccak-f permutation
    pub inner: &'a mut KeccakPermCols<T>,
    /// Columns for sponge and padding
    pub sponge: &'a mut KeccakSpongeCols<T>,
    /// Columns for opcode interface and operand memory access
    pub opcode: &'a mut KeccakOpcodeCols<T>,
    /// Auxiliary columns for offline memory checking
    /// This should be convertable to [KeccakMemoryCols]
    pub mem_oc: &'a mut [T],
}

/// Columns specific to the KECCAK256 opcode.
/// The opcode instruction format is (a, b, len, d, e, f)
#[allow(clippy::too_many_arguments)]
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, AlignedBorrow, derive_new::new)]
pub struct KeccakOpcodeCols<T> {
    /// Program counter
    pub pc: T,
    /// True for all rows that are part of opcode execution.
    /// False on dummy rows only used to pad the height.
    pub is_enabled: T,
    /// The starting timestamp to use for memory access in this row.
    /// A single row will do multiple memory accesses.
    pub start_timestamp: T,
    // Operands:
    pub a: T,
    pub b: T,
    pub c: T,
    pub d: T,
    pub e: T,
    pub f: T,
    // Memory values
    /// dst <- [a]_d
    pub dst: T,
    /// src <- [b]_d
    pub src: T,
    /// The remaining length of the unpadded input, in bytes.
    /// If this row is receiving from opcode bus, then
    /// len <- [c]_f
    pub len: T,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, AlignedBorrow)]
pub struct KeccakSpongeCols<T> {
    /// Only used on first row of a round to determine whether the state
    /// prior to absorb should be reset to all 0s.
    /// Constrained to be zero if not first round.
    pub is_new_start: T,

    /// Whether the current byte is a padding byte.
    ///
    /// If this row represents a full input block, this should contain all 0s.
    pub is_padding_byte: [T; KECCAK_RATE_BYTES],

    /// The block being absorbed, which may contain input bytes and padding
    /// bytes.
    pub block_bytes: [T; KECCAK_RATE_BYTES],

    /// For each of the first [KECCAK_RATE_U16S] `u16` limbs in the state,
    /// the most significant byte of the limb.
    /// Here `state` is the postimage state if last round and the preimage
    /// state if first round. It can be junk if not first or last round.
    pub state_hi: [T; KECCAK_RATE_U16S],
}

// Grouping all memory aux columns together because they can't use AlignedBorrow
#[derive(Clone, Debug)]
pub struct KeccakMemoryCols<T> {
    pub op_reads: [MemoryReadAuxCols<1, T>; KECCAK_EXECUTION_READS],
    // TODO[jpw] switch to word_size=8
    pub absorb_reads: [MemoryReadAuxCols<1, T>; KECCAK_ABSORB_READS],
    // TODO[jpw] switch to word_size=? (4 or 8 or 16)
    pub digest_writes: [MemoryWriteAuxCols<1, T>; KECCAK_DIGEST_WRITES],
}

impl<'a, T: Copy> KeccakVmColsRef<'a, T> {
    pub const fn remaining_len(&self) -> T {
        self.opcode.len
    }

    pub const fn is_new_start(&self) -> T {
        self.sponge.is_new_start
    }

    pub fn postimage(&self, y: usize, x: usize, limb: usize) -> T {
        // WARNING: once plonky3 commit is updated this needs to be changed to y, x
        self.inner.a_prime_prime_prime(x, y, limb)
    }

    pub fn is_first_round(&self) -> T {
        *self.inner.step_flags.first().unwrap()
    }

    pub fn is_last_round(&self) -> T {
        *self.inner.step_flags.last().unwrap()
    }
}

impl<'a, T> KeccakVmColsRef<'a, T> {
    pub fn from_slice(slc: &'a [T]) -> Self {
        let (inner, slc) = slc.split_at(NUM_KECCAK_PERM_COLS);
        let (sponge, slc) = slc.split_at(NUM_KECCAK_SPONGE_COLS);
        let (opcode, mem_oc) = slc.split_at(NUM_KECCAK_OPCODE_COLS);
        Self {
            inner: inner.borrow(),
            sponge: sponge.borrow(),
            opcode: opcode.borrow(),
            mem_oc,
        }
    }
}

impl<'a, T> KeccakVmColsMut<'a, T> {
    pub fn from_mut_slice(slc: &'a mut [T]) -> Self {
        let (inner, slc) = slc.split_at_mut(NUM_KECCAK_PERM_COLS);
        let (sponge, slc) = slc.split_at_mut(NUM_KECCAK_SPONGE_COLS);
        let (opcode, mem_oc) = slc.split_at_mut(NUM_KECCAK_OPCODE_COLS);
        Self {
            inner: inner.borrow_mut(),
            sponge: sponge.borrow_mut(),
            opcode: opcode.borrow_mut(),
            mem_oc,
        }
    }
}

impl<T: Copy> KeccakOpcodeCols<T> {
    pub fn assert_eq<AB: AirBuilder>(&self, builder: &mut AB, other: Self)
    where
        T: Into<AB::Expr>,
    {
        builder.assert_eq(self.is_enabled, other.is_enabled);
        builder.assert_eq(self.start_timestamp, other.start_timestamp);
        builder.assert_eq(self.a, other.a);
        builder.assert_eq(self.b, other.b);
        builder.assert_eq(self.c, other.c);
        builder.assert_eq(self.d, other.d);
        builder.assert_eq(self.e, other.e);
        builder.assert_eq(self.dst, other.dst);
        builder.assert_eq(self.src, other.src);
        builder.assert_eq(self.len, other.len);
    }
}

pub const NUM_KECCAK_OPCODE_COLS: usize = size_of::<KeccakOpcodeCols<u8>>();
pub const NUM_KECCAK_SPONGE_COLS: usize = size_of::<KeccakSpongeCols<u8>>();

impl<T> KeccakMemoryCols<T> {
    pub fn width(mem_oc: &MemoryOfflineChecker) -> usize {
        (KECCAK_EXECUTION_READS + KECCAK_ABSORB_READS) * MemoryReadAuxCols::<1, T>::width(mem_oc)
            + KECCAK_DIGEST_WRITES * MemoryWriteAuxCols::<1, T>::width(mem_oc)
    }

    pub fn from_slice(slc: &[T], mem_oc: &MemoryOfflineChecker) -> Self
    where
        T: Clone,
    {
        let mut it = slc.iter().cloned();
        let mut next = || MemoryReadAuxCols::from_iterator(&mut it, &mem_oc.timestamp_lt_air);
        let op_reads = from_fn(|_| next());
        let absorb_reads = from_fn(|_| next());
        let digest_writes =
            from_fn(|_| MemoryWriteAuxCols::from_iterator(&mut it, &mem_oc.timestamp_lt_air));

        Self {
            op_reads,
            absorb_reads,
            digest_writes,
        }
    }

    pub fn flatten(self) -> Vec<T> {
        self.op_reads
            .into_iter()
            .flat_map(|read| read.flatten())
            .chain(
                self.absorb_reads
                    .into_iter()
                    .flat_map(|read| read.flatten()),
            )
            .chain(
                self.digest_writes
                    .into_iter()
                    .flat_map(|write| write.flatten()),
            )
            .collect()
    }
}
