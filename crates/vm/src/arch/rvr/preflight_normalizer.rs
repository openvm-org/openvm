//! Build OpenVM memory records from self-contained rvr preflight logs.
//!
//! R1: the C tracer emits self-contained events (each carries `prev_timestamp`
//! from the timestamp shadow and `prev_value`, the block's pre-access value) and
//! records first-touched blocks in a `touched` buffer. This module therefore
//! does two cheap linear passes — build the per-access aux vector, and finalize
//! `touched_memory` from the touched-block list + shadow + live memory — with no
//! log replay, sort, or per-access map.

use std::array;

use openvm_instructions::riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS};
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

/// Reusable per-address-space bitmaps for emitting touched blocks directly in canonical address
/// order. One bit represents one [`WORD_BYTES`]-sized block.
#[derive(Debug, Default)]
pub(crate) struct TouchedOrderScratch {
    register: Vec<u64>,
    memory: Vec<u64>,
    public_values: Vec<u64>,
}

impl TouchedOrderScratch {
    pub(crate) fn prepare(
        &mut self,
        register_blocks: usize,
        memory_blocks: usize,
        public_values_blocks: usize,
    ) {
        prepare_bitmap(&mut self.register, register_blocks);
        prepare_bitmap(&mut self.memory, memory_blocks);
        prepare_bitmap(&mut self.public_values, public_values_blocks);
    }

    fn bitmap_mut(&mut self, addr_space: u32) -> &mut [u64] {
        if addr_space == RV64_REGISTER_AS {
            &mut self.register
        } else if addr_space == PUBLIC_VALUES_AS {
            &mut self.public_values
        } else {
            &mut self.memory
        }
    }
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
    build_access_aux: bool,
) -> Result<PreflightMemoryReplay<F>, PreflightNormalizeError> {
    build_preflight_replay_with_scratch(
        memory,
        shadows,
        touched,
        logs,
        build_access_aux,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_preflight_replay_with_scratch<F: Field>(
    memory: &GuestMemory,
    shadows: &PreflightShadowsView,
    touched: &[TouchedBlock],
    logs: &[MemoryLogEntry],
    build_access_aux: bool,
    access_aux_backing: Option<Vec<PreflightMemoryAccessAux<F>>>,
    touched_order: Option<&mut TouchedOrderScratch>,
) -> Result<PreflightMemoryReplay<F>, PreflightNormalizeError> {
    let detailed_profile =
        std::env::var("OPENVM_RVR_PREFLIGHT_PROFILE_DETAIL").as_deref() == Ok("1");
    let replay_started = std::time::Instant::now();
    let access_aux = if build_access_aux && access_aux_backing.is_none() {
        logs.iter()
            .enumerate()
            .map(|(index, entry)| normalize_access(memory, index, entry))
            .collect::<Result<Vec<_>, _>>()?
    } else if build_access_aux {
        let mut access_aux = access_aux_backing.unwrap_or_default();
        access_aux.clear();
        access_aux.reserve(logs.len());
        for (index, entry) in logs.iter().enumerate() {
            access_aux.push(normalize_access(memory, index, entry)?);
        }
        access_aux
    } else {
        access_aux_backing.unwrap_or_default()
    };
    let access_aux_finished = std::time::Instant::now();

    // Finalize touched_memory from the touched-block list: final value from live
    // memory, final timestamp from the shadow. Sorted by (addr_space, block_ptr)
    // to match `TracingMemory::touched_blocks_to_equipartition`.
    let ordered_touched = touched_order.is_some();
    let mut touched_memory: TouchedMemory<F> = if let Some(touched_order) = touched_order {
        build_ordered_touched_memory(memory, shadows, touched, touched_order)
    } else {
        touched
            .iter()
            .map(|tb| touched_memory_entry(memory, shadows, tb.addr_space, tb.block_addr as usize))
            .collect()
    };
    let touched_collect_finished = std::time::Instant::now();
    if !ordered_touched {
        touched_memory.sort_unstable_by_key(|(addr, _)| *addr);
    }
    let touched_sort_finished = std::time::Instant::now();

    if detailed_profile {
        eprintln!(
            "OPENVM_RVR_REPLAY_DETAIL access_aux_us={} touched_collect_us={} touched_sort_us={} \
             access_aux_required={} access_records={} touched_blocks={}",
            (access_aux_finished - replay_started).as_micros(),
            (touched_collect_finished - access_aux_finished).as_micros(),
            (touched_sort_finished - touched_collect_finished).as_micros(),
            build_access_aux as u8,
            logs.len(),
            touched.len(),
        );
    }

    Ok(PreflightMemoryReplay {
        touched_memory,
        access_aux,
    })
}

#[inline(always)]
fn normalize_access<F: Field>(
    memory: &GuestMemory,
    index: usize,
    entry: &MemoryLogEntry,
) -> Result<PreflightMemoryAccessAux<F>, PreflightNormalizeError> {
    let addr_space = entry.addr_space as u32;
    let config = memory
        .memory
        .config
        .get(addr_space as usize)
        .ok_or(PreflightNormalizeError::InvalidAddressSpace { index, addr_space })?;
    let cell_size = config.layout.size();
    let block_bytes = BLOCK_FE_WIDTH * cell_size;
    let entry_address =
        usize::try_from(entry.address).map_err(|_| PreflightNormalizeError::AddressOutOfRange {
            index,
            address: entry.address,
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
        prev_data: block_bytes_to_fields(memory, addr_space, &entry.prev_value.to_le_bytes()),
    })
}

fn prepare_bitmap(bitmap: &mut Vec<u64>, blocks: usize) {
    bitmap.resize(blocks.div_ceil(u64::BITS as usize), 0);
    bitmap.fill(0);
}

fn build_ordered_touched_memory<F: Field>(
    memory: &GuestMemory,
    shadows: &PreflightShadowsView,
    touched: &[TouchedBlock],
    scratch: &mut TouchedOrderScratch,
) -> TouchedMemory<F> {
    for block in touched {
        let block_idx = block.block_addr as usize / WORD_BYTES;
        let bitmap = scratch.bitmap_mut(block.addr_space);
        bitmap[block_idx / u64::BITS as usize] |= 1u64 << (block_idx % u64::BITS as usize);
    }

    let TouchedOrderScratch {
        register,
        memory: memory_bitmap,
        public_values,
    } = scratch;
    let mut spaces = [
        (RV64_REGISTER_AS, register),
        (RV64_MEMORY_AS, memory_bitmap),
        (PUBLIC_VALUES_AS, public_values),
    ];
    spaces.sort_unstable_by_key(|(addr_space, _)| *addr_space);

    let mut ordered = Vec::with_capacity(touched.len());
    for (addr_space, bitmap) in spaces {
        for (word_idx, word) in bitmap.iter_mut().enumerate() {
            let mut pending = *word;
            *word = 0;
            while pending != 0 {
                let bit = pending.trailing_zeros() as usize;
                let block_idx = word_idx * u64::BITS as usize + bit;
                ordered.push(touched_memory_entry(
                    memory,
                    shadows,
                    addr_space,
                    block_idx * WORD_BYTES,
                ));
                pending &= pending - 1;
            }
        }
    }
    debug_assert_eq!(ordered.len(), touched.len());
    ordered
}

fn touched_memory_entry<F: Field>(
    memory: &GuestMemory,
    shadows: &PreflightShadowsView,
    addr_space: u32,
    aligned_byte_addr: usize,
) -> ((u32, u32), TimestampedValues<F, BLOCK_FE_WIDTH>) {
    let config = &memory.memory.config[addr_space as usize];
    let cell_size = config.layout.size();
    let block_bytes = BLOCK_FE_WIDTH * cell_size;
    debug_assert_eq!(
        block_bytes, WORD_BYTES,
        "rvr preflight shadow assumes WORD_BYTES blocks (U16 cells)"
    );
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

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;

    use super::*;
    use crate::{system::memory::AddressMap, utils::test_system_config};

    #[test]
    fn pooled_replay_and_bitmap_order_are_byte_equal_to_baseline() {
        let config = test_system_config();
        let mut memory = GuestMemory::new(AddressMap::from_mem_config(&config.memory_config));
        unsafe {
            memory.write::<u16, BLOCK_FE_WIDTH>(RV64_REGISTER_AS, 4, [1, 2, 3, 4]);
            memory.write::<u16, BLOCK_FE_WIDTH>(RV64_MEMORY_AS, 0, [5, 6, 7, 8]);
            memory.write::<u16, BLOCK_FE_WIDTH>(RV64_MEMORY_AS, 16, [9, 10, 11, 12]);
            memory.write::<u16, BLOCK_FE_WIDTH>(PUBLIC_VALUES_AS, 0, [13, 14, 15, 16]);
        }

        let touched = vec![
            TouchedBlock {
                addr_space: RV64_MEMORY_AS,
                block_addr: 32,
                initial_value: 0,
            },
            TouchedBlock {
                addr_space: PUBLIC_VALUES_AS,
                block_addr: 0,
                initial_value: 0,
            },
            TouchedBlock {
                addr_space: RV64_REGISTER_AS,
                block_addr: 8,
                initial_value: 0,
            },
            TouchedBlock {
                addr_space: RV64_MEMORY_AS,
                block_addr: 0,
                initial_value: 0,
            },
        ];
        let mut register_shadow = vec![0; 64];
        let mut memory_shadow = vec![0; 64];
        let mut public_values_shadow = vec![0; 64];
        register_shadow[1] = 11;
        memory_shadow[0] = 22;
        memory_shadow[4] = 33;
        public_values_shadow[0] = 44;
        let shadows = PreflightShadowsView {
            register: &register_shadow,
            memory: &memory_shadow,
            public_values: &public_values_shadow,
        };
        let logs = vec![
            MemoryLogEntry {
                timestamp: 7,
                prev_timestamp: 2,
                addr_space: RV64_MEMORY_AS as u8,
                width: WORD_BYTES as u8,
                address: 32,
                prev_value: 0x0807_0605_0403_0201,
                ..Default::default()
            },
            MemoryLogEntry {
                timestamp: 9,
                prev_timestamp: 0,
                addr_space: RV64_REGISTER_AS as u8,
                width: WORD_BYTES as u8,
                address: 8,
                prev_value: 0x100f_0e0d_0c0b_0a09,
                ..Default::default()
            },
        ];

        let baseline =
            build_preflight_replay::<BabyBear>(&memory, &shadows, &touched, &logs, true).unwrap();
        let mut scratch = TouchedOrderScratch::default();
        scratch.prepare(
            register_shadow.len(),
            memory_shadow.len(),
            public_values_shadow.len(),
        );
        let candidate = build_preflight_replay_with_scratch::<BabyBear>(
            &memory,
            &shadows,
            &touched,
            &logs,
            true,
            Some(Vec::with_capacity(logs.len())),
            Some(&mut scratch),
        )
        .unwrap();

        assert_eq!(candidate, baseline);
        assert!(scratch.register.iter().all(|&word| word == 0));
        assert!(scratch.memory.iter().all(|&word| word == 0));
        assert!(scratch.public_values.iter().all(|&word| word == 0));
    }
}
