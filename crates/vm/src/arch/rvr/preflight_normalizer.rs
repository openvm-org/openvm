//! Replay rvr preflight memory logs into OpenVM memory records.

use std::{array, collections::BTreeMap};

use openvm_stark_backend::p3_field::Field;

use super::preflight::{
    MemoryLogEntry, PREFLIGHT_MEMORY_KIND_READ, PREFLIGHT_MEMORY_KIND_TOUCH,
    PREFLIGHT_MEMORY_KIND_WRITE,
};
use crate::{
    arch::{AddressSpaceHostLayout, BLOCK_FE_WIDTH},
    system::{
        memory::{online::GuestMemory, TimestampedValues},
        TouchedMemory,
    },
};

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
    #[error("memory log entry {index} has unsupported kind {kind}")]
    UnsupportedKind { index: usize, kind: u8 },
    #[error("memory log entry {index} width {width} exceeds block size {block_bytes}")]
    WidthExceedsBlock {
        index: usize,
        width: usize,
        block_bytes: usize,
    },
    #[error("memory log entry {index} range [{start}, {end}) exceeds block size {block_bytes}")]
    AccessCrossesBlock {
        index: usize,
        start: usize,
        end: usize,
        block_bytes: usize,
    },
    #[error("memory log entry {index} address {address:#x} exceeds OpenVM pointer range")]
    AddressOutOfRange { index: usize, address: u64 },
}

#[derive(Clone)]
struct BlockState {
    timestamp: u32,
    bytes: Vec<u8>,
}

/// Replay byte-addressed rvr memory events into OpenVM block-timestamped memory.
///
/// `MemoryLogEntry.address` is a byte pointer in the address space's host
/// layout. The returned touched-memory pointer is the AS-native cell pointer
/// used by `TracingMemory::finalize()`.
pub fn normalize_preflight_memory_logs<F: Field>(
    initial_memory: &GuestMemory,
    logs: &[MemoryLogEntry],
) -> Result<PreflightMemoryReplay<F>, PreflightNormalizeError> {
    let mut indexed_logs = logs.iter().copied().enumerate().collect::<Vec<_>>();
    indexed_logs.sort_by_key(|(index, entry)| (entry.timestamp, *index));

    let mut blocks = BTreeMap::<(u32, u32), BlockState>::new();
    let mut aux_by_log = (0..logs.len()).map(|_| None).collect::<Vec<_>>();

    for (index, entry) in indexed_logs {
        let addr_space = entry.addr_space as u32;
        let config = initial_memory
            .memory
            .config
            .get(addr_space as usize)
            .ok_or(PreflightNormalizeError::InvalidAddressSpace { index, addr_space })?;
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
        let block_key = (addr_space, block_ptr);

        let block = match blocks.entry(block_key) {
            std::collections::btree_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::btree_map::Entry::Vacant(vacant) => {
                let bytes = unsafe {
                    initial_memory
                        .memory
                        .get_u8_slice(addr_space, aligned_byte_addr, block_bytes)
                        .to_vec()
                };
                vacant.insert(BlockState {
                    timestamp: 0,
                    bytes,
                })
            }
        };

        let prev_timestamp = block.timestamp;
        let prev_data = block_bytes_to_fields(initial_memory, addr_space, &block.bytes);
        aux_by_log[index] = Some(PreflightMemoryAccessAux {
            log_index: index,
            entry,
            block_addr: block_key,
            prev_timestamp,
            prev_data,
        });

        match entry.kind {
            PREFLIGHT_MEMORY_KIND_READ | PREFLIGHT_MEMORY_KIND_TOUCH => {}
            PREFLIGHT_MEMORY_KIND_WRITE => {
                let width = entry.width as usize;
                if width > block_bytes {
                    return Err(PreflightNormalizeError::WidthExceedsBlock {
                        index,
                        width,
                        block_bytes,
                    });
                }
                let start = entry_address - aligned_byte_addr;
                let end = start + width;
                if end > block_bytes {
                    return Err(PreflightNormalizeError::AccessCrossesBlock {
                        index,
                        start,
                        end,
                        block_bytes,
                    });
                }
                block.bytes[start..end].copy_from_slice(&entry.value.to_le_bytes()[..width]);
            }
            kind => {
                return Err(PreflightNormalizeError::UnsupportedKind { index, kind });
            }
        }
        // OpenVM's online memory updates block metadata on every access, not
        // just writes. A trailing read therefore determines touched_memory's
        // timestamp while preserving the last written value.
        block.timestamp = entry.timestamp;
    }

    let touched_memory = blocks
        .into_iter()
        .filter_map(|(addr, block)| {
            (block.timestamp > 0).then(|| {
                let values = block_bytes_to_fields(initial_memory, addr.0, &block.bytes);
                (
                    addr,
                    TimestampedValues {
                        timestamp: block.timestamp,
                        values,
                    },
                )
            })
        })
        .collect();
    let access_aux = aux_by_log
        .into_iter()
        .map(|aux| aux.expect("one aux record is produced for every memory log entry"))
        .collect();

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
