use std::{collections::BTreeSet, mem::size_of};

use elf::{endian::LittleEndian, ElfBytes};

// Format: https://llvm.org/docs/Extensions.html#sht-llvm-bb-addr-map-section-basic-block-address-map
// LLVM assigns this OS-specific ELF type, so maps can be identified without a section name.
const SHT_LLVM_BB_ADDR_MAP: u32 = 0x6fff_4c0a;
// Wire format emitted by the pinned LLVM 21 guest toolchain.
const LLVM_BB_ADDR_MAP_VERSION: u8 = 3;
// These bits determine which variable-length fields the parser must consume.
// A function execution count follows its address ranges.
const LLVM_BB_ADDR_MAP_FUNC_ENTRY_COUNT: u8 = 1 << 0;
// Each block has a relative execution frequency.
const LLVM_BB_ADDR_MAP_BB_FREQ: u8 = 1 << 1;
// Each block has successor IDs and branch probabilities.
const LLVM_BB_ADDR_MAP_BRANCH_PROB: u8 = 1 << 2;
// A function may have multiple discontiguous address ranges.
const LLVM_BB_ADDR_MAP_MULTI_RANGE: u8 = 1 << 3;
// Ranges omit their individual block records.
const LLVM_BB_ADDR_MAP_OMIT_ENTRIES: u8 = 1 << 4;
// Block sizes are split around encoded callsite offsets.
const LLVM_BB_ADDR_MAP_CALLSITES: u8 = 1 << 5;
const LLVM_BB_ADDR_MAP_SUPPORTED_FEATURES: u8 = (1 << 6) - 1;
// LLVM 21 defines five boolean flags in each block's metadata field.
const LLVM_BB_ADDR_MAP_METADATA_MASK: u64 = (1 << 5) - 1;

fn read_byte(bytes: &mut &[u8]) -> Option<u8> {
    let (&byte, rest) = bytes.split_first()?;
    *bytes = rest;
    Some(byte)
}

fn read_u64(bytes: &mut &[u8]) -> Option<u64> {
    let (value, rest) = bytes.split_at_checked(size_of::<u64>())?;
    *bytes = rest;
    Some(u64::from_le_bytes(value.try_into().ok()?))
}

fn read_uleb128(bytes: &mut &[u8]) -> Option<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = read_byte(bytes)?;
        let digit = u64::from(byte & 0x7f);
        if shift == 63 && digit > 1 {
            return None;
        }
        value |= digit << shift;
        if byte & 0x80 == 0 {
            return Some(value);
        }
    }
    None
}

fn read_uleb32(bytes: &mut &[u8]) -> Option<u32> {
    read_uleb128(bytes)?.try_into().ok()
}

/// Decodes the LLVM 21 basic-block address map emitted by the OpenVM guest toolchain.
fn decode(mut bytes: &[u8]) -> Option<BTreeSet<u32>> {
    let mut block_starts = BTreeSet::new();
    while !bytes.is_empty() {
        if read_byte(&mut bytes)? != LLVM_BB_ADDR_MAP_VERSION {
            return None;
        }
        let features = read_byte(&mut bytes)?;
        if features & !LLVM_BB_ADDR_MAP_SUPPORTED_FEATURES != 0 {
            return None;
        }

        let range_count = if features & LLVM_BB_ADDR_MAP_MULTI_RANGE != 0 {
            let count = read_uleb32(&mut bytes)?;
            (count != 0).then_some(count)?
        } else {
            1
        };
        let mut total_blocks = 0u64;
        for _ in 0..range_count {
            let range_start = read_u64(&mut bytes)?;
            block_starts.insert(range_start.try_into().ok()?);
            let block_count = read_uleb32(&mut bytes)?;
            if features & LLVM_BB_ADDR_MAP_OMIT_ENTRIES != 0 {
                continue;
            }
            total_blocks = total_blocks.checked_add(u64::from(block_count))?;

            let mut previous_end = range_start;
            for _ in 0..block_count {
                let _id = read_uleb32(&mut bytes)?;
                let block_start = previous_end.checked_add(u64::from(read_uleb32(&mut bytes)?))?;
                let mut cursor = block_start;
                if features & LLVM_BB_ADDR_MAP_CALLSITES != 0 {
                    for _ in 0..read_uleb32(&mut bytes)? {
                        cursor = cursor.checked_add(u64::from(read_uleb32(&mut bytes)?))?;
                    }
                }
                previous_end = cursor.checked_add(u64::from(read_uleb32(&mut bytes)?))?;
                let metadata = read_uleb128(&mut bytes)?;
                if metadata & !LLVM_BB_ADDR_MAP_METADATA_MASK != 0 {
                    return None;
                }

                if let Ok(pc) = u32::try_from(block_start) {
                    block_starts.insert(pc);
                }
            }
        }

        if features & LLVM_BB_ADDR_MAP_FUNC_ENTRY_COUNT != 0 {
            read_uleb128(&mut bytes)?;
        }
        if features & (LLVM_BB_ADDR_MAP_BB_FREQ | LLVM_BB_ADDR_MAP_BRANCH_PROB) != 0 {
            for _ in 0..total_blocks {
                if features & LLVM_BB_ADDR_MAP_BB_FREQ != 0 {
                    read_uleb128(&mut bytes)?;
                }
                if features & LLVM_BB_ADDR_MAP_BRANCH_PROB != 0 {
                    for _ in 0..read_uleb32(&mut bytes)? {
                        read_uleb32(&mut bytes)?;
                        read_uleb32(&mut bytes)?;
                    }
                }
            }
        }
    }
    Some(block_starts)
}

pub(super) fn block_starts(elf: &ElfBytes<'_, LittleEndian>) -> BTreeSet<u32> {
    let mut block_starts = BTreeSet::new();
    if let Some(sections) = elf.section_headers() {
        for section in sections
            .iter()
            .filter(|section| section.sh_type == SHT_LLVM_BB_ADDR_MAP)
        {
            let Ok((bytes, None)) = elf.section_data(&section) else {
                continue;
            };
            if let Some(starts) = decode(bytes) {
                block_starts.extend(starts);
            }
        }
    }
    block_starts
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_basic_block_address_map() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_CALLSITES];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks
            0, 0, 0, 4, 8, // block 0: [0x100, 0x104)
            1, 4, 1, 4, 4, 1, // block 1: [0x108, 0x110), one callsite at 0x10c
        ]);

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100, 0x108])));
    }

    #[test]
    fn rejects_truncated_basic_block_address_map() {
        assert_eq!(decode(&[LLVM_BB_ADDR_MAP_VERSION]), None);
    }

    #[test]
    fn omitted_entries_do_not_iterate_declared_block_count() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_OMIT_ENTRIES];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0x0f]);

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100])));
    }
}
