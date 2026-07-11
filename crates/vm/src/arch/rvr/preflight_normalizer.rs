//! Build OpenVM memory records from self-contained rvr preflight logs.
//!
//! R1: the C tracer emits self-contained events (each carries `prev_timestamp`
//! from the timestamp shadow and `prev_value`, the block's pre-access value) and
//! records first-touched blocks in a `touched` buffer. This module therefore
//! does two cheap linear passes — build the per-access aux vector, and finalize
//! `touched_memory` from the touched-block list + shadow + live memory — with no
//! log replay, sort, or per-access map.

use std::array;

use openvm_instructions::riscv::RV64_REGISTER_AS;
use openvm_stark_backend::p3_field::Field;

use super::preflight::{MemoryLogEntry, TouchedBlock};
use crate::{
    arch::{AddressSpaceHostLayout, BLOCK_FE_WIDTH},
    system::{
        memory::{merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory, TimestampedValues},
        TouchedMemory,
    },
};

/// Traced blocks are `WORD_BYTES` bytes; the C shadow indexes by
/// `block_addr / WORD_BYTES`. All rvr-traced address spaces (register, memory,
/// public values) use U16 cells, so this equals `BLOCK_FE_WIDTH` cells.
pub(crate) const WORD_BYTES: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightMemoryAccessAux<F> {
    pub log_index: usize,
    pub entry: MemoryLogEntry,
    pub block_addr: (u32, u32),
    pub prev_timestamp: u32,
    pub prev_data: [F; BLOCK_FE_WIDTH],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreflightMemoryReplay<F> {
    pub touched_memory: TouchedMemory<F>,
    pub access_aux: Vec<PreflightMemoryAccessAux<F>>,
}

#[derive(Debug, thiserror::Error)]
pub enum PreflightNormalizeError {
    #[error("memory log entry {index} has invalid address space {addr_space}")]
    InvalidAddressSpace { index: usize, addr_space: u32 },
    #[error("memory log entry {index} address {address:#x} exceeds OpenVM pointer range")]
    AddressOutOfRange { index: usize, address: u64 },
}

/// Read-only view of the per-address-space timestamp shadows, used to recover
/// each touched block's final-access timestamp.
pub struct PreflightShadowsView<'a> {
    pub register: &'a [u32],
    pub memory: &'a [u32],
    pub public_values: &'a [u32],
}

impl PreflightShadowsView<'_> {
    #[inline]
    fn final_timestamp(&self, addr_space: u32, block_idx: usize) -> u32 {
        let shadow = if addr_space == RV64_REGISTER_AS {
            self.register
        } else if addr_space == PUBLIC_VALUES_AS {
            self.public_values
        } else {
            self.memory
        };
        shadow.get(block_idx).copied().unwrap_or(0)
    }
}

/// Build the per-access aux vector and the finalized `touched_memory` from the
/// self-contained preflight log and touched-block buffer.
///
/// `memory` is the post-execution live memory (final block values); `shadows`
/// supplies each block's final-access timestamp; `touched` lists every block
/// first-touched this segment (in first-touch order); `logs` is the
/// self-contained, already timestamp-ordered memory log.
pub fn build_preflight_replay<F: Field>(
    memory: &GuestMemory,
    shadows: &PreflightShadowsView,
    touched: &[TouchedBlock],
    logs: &[MemoryLogEntry],
) -> Result<PreflightMemoryReplay<F>, PreflightNormalizeError> {
    let access_aux =
        logs.iter()
            .enumerate()
            .map(|(index, entry)| {
                let addr_space = entry.addr_space as u32;
                let config =
                    memory.memory.config.get(addr_space as usize).ok_or(
                        PreflightNormalizeError::InvalidAddressSpace { index, addr_space },
                    )?;
                let cell_size = config.layout.size();
                let block_bytes = BLOCK_FE_WIDTH * cell_size;
                let entry_address = usize::try_from(entry.address).map_err(|_| {
                    PreflightNormalizeError::AddressOutOfRange {
                        index,
                        address: entry.address,
                    }
                })?;
                let aligned_byte_addr = (entry_address / block_bytes) * block_bytes;
                let block_ptr = u32::try_from(aligned_byte_addr / cell_size).map_err(|_| {
                    PreflightNormalizeError::AddressOutOfRange {
                        index,
                        address: entry.address,
                    }
                })?;
                Ok(PreflightMemoryAccessAux {
                    log_index: index,
                    entry: *entry,
                    block_addr: (addr_space, block_ptr),
                    prev_timestamp: entry.prev_timestamp,
                    prev_data: block_bytes_to_fields(
                        memory,
                        addr_space,
                        &entry.prev_value.to_le_bytes(),
                    ),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

    // Finalize touched_memory from the touched-block list: final value from live
    // memory, final timestamp from the shadow. Sorted by (addr_space, block_ptr)
    // to match `TracingMemory::touched_blocks_to_equipartition`.
    let mut touched_memory: TouchedMemory<F> = touched
        .iter()
        .map(|tb| {
            let addr_space = tb.addr_space;
            let config = &memory.memory.config[addr_space as usize];
            let cell_size = config.layout.size();
            let block_bytes = BLOCK_FE_WIDTH * cell_size;
            debug_assert_eq!(
                block_bytes, WORD_BYTES,
                "rvr preflight shadow assumes WORD_BYTES blocks (U16 cells)"
            );
            let aligned_byte_addr = tb.block_addr as usize;
            let block_ptr = (aligned_byte_addr / cell_size) as u32;
            let block_idx = aligned_byte_addr / WORD_BYTES;
            let timestamp = shadows.final_timestamp(addr_space, block_idx);
            let bytes = unsafe {
                memory
                    .memory
                    .get_u8_slice(addr_space, aligned_byte_addr, block_bytes)
            };
            let values = block_bytes_to_fields(memory, addr_space, bytes);
            (
                (addr_space, block_ptr),
                TimestampedValues { timestamp, values },
            )
        })
        .collect();
    touched_memory.sort_unstable_by_key(|(addr, _)| *addr);

    Ok(PreflightMemoryReplay {
        touched_memory,
        access_aux,
    })
}

fn block_bytes_to_fields<F: Field>(
    memory: &GuestMemory,
    addr_space: u32,
    bytes: &[u8],
) -> [F; BLOCK_FE_WIDTH] {
    let layout = &memory.memory.config[addr_space as usize].layout;
    let cell_size = layout.size();
    array::from_fn(|i| {
        let start = i * cell_size;
        let end = start + cell_size;
        unsafe { layout.to_field(&bytes[start..end]) }
    })
}
