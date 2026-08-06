use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
};

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
// Returns and tail calls leave the local machine CFG; function entries and callsites cover them.
const LLVM_BB_ADDR_MAP_HAS_RETURN: u64 = 1 << 0;
const LLVM_BB_ADDR_MAP_HAS_TAIL_CALL: u64 = 1 << 1;
// The source block ends in an indirect branch.
const LLVM_BB_ADDR_MAP_HAS_INDIRECT_BRANCH: u64 = 1 << 4;
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
        if features & LLVM_BB_ADDR_MAP_OMIT_ENTRIES != 0
            && features & (LLVM_BB_ADDR_MAP_BB_FREQ | LLVM_BB_ADDR_MAP_BRANCH_PROB) != 0
        {
            return None;
        }

        let range_count = if features & LLVM_BB_ADDR_MAP_MULTI_RANGE != 0 {
            let count = read_uleb32(&mut bytes)?;
            (count != 0).then_some(count)?
        } else {
            1
        };
        let mut function_block_starts = BTreeMap::new();
        let mut block_ids = Vec::new();
        let mut computed_branch_ids = BTreeSet::new();
        for range_idx in 0..range_count {
            let range_start = read_u64(&mut bytes)?;
            if range_idx == 0 {
                block_starts.insert(range_start.try_into().ok()?);
            }
            let block_count = read_uleb32(&mut bytes)?;
            if features & LLVM_BB_ADDR_MAP_OMIT_ENTRIES != 0 {
                continue;
            }
            let mut previous_end = range_start;
            for _ in 0..block_count {
                let id = read_uleb32(&mut bytes)?;
                block_ids.push(id);
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
                if metadata & LLVM_BB_ADDR_MAP_HAS_INDIRECT_BRANCH != 0
                    && metadata & (LLVM_BB_ADDR_MAP_HAS_RETURN | LLVM_BB_ADDR_MAP_HAS_TAIL_CALL)
                        == 0
                {
                    computed_branch_ids.insert(id);
                }

                let pc = block_start.try_into().ok()?;
                if function_block_starts.insert(id, pc).is_some() {
                    return None;
                }
            }
        }
        if features & LLVM_BB_ADDR_MAP_FUNC_ENTRY_COUNT != 0 {
            read_uleb128(&mut bytes)?;
        }
        if features & (LLVM_BB_ADDR_MAP_BB_FREQ | LLVM_BB_ADDR_MAP_BRANCH_PROB) != 0 {
            for id in block_ids {
                if features & LLVM_BB_ADDR_MAP_BB_FREQ != 0 {
                    read_uleb128(&mut bytes)?;
                }
                if features & LLVM_BB_ADDR_MAP_BRANCH_PROB != 0 {
                    for _ in 0..read_uleb32(&mut bytes)? {
                        let successor_id = read_uleb32(&mut bytes)?;
                        read_uleb32(&mut bytes)?;
                        let successor_pc = *function_block_starts.get(&successor_id)?;
                        if computed_branch_ids.contains(&id) {
                            block_starts.insert(successor_pc);
                        }
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

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100])));
    }

    #[test]
    fn branch_probabilities_retain_only_indirect_successors() {
        let features = LLVM_BB_ADDR_MAP_MULTI_RANGE | LLVM_BB_ADDR_MAP_BRANCH_PROB;
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, features, 2];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks in range 1
            0, 0, 4, 0, // block 0: [0x100, 0x104)
            4, 4, 4, 16, // block 4: [0x108, 0x10c), indirect branch
        ]);
        bytes.extend_from_slice(&0x200u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks in range 2
            8, 0, 4, 0, // block 8: [0x200, 0x204)
            12, 4, 4, 0, // block 12: [0x208, 0x20c)
            1, 4, 1, // block 0 -> block 4
            1, 12, 1, // indirect block 4 -> block 12
            1, 12, 1, // block 8 -> block 12
            0, // block 12 has no successors
        ]);

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100, 0x208])));
    }

    #[test]
    fn omits_interior_blocks_without_indirect_control_flow() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, 0];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks
            0, 0, 4, 8, // block 0: [0x100, 0x104)
            1, 4, 4, 1, // block 1: [0x108, 0x10c)
        ]);

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100])));
    }

    #[test]
    fn ignores_return_and_tail_call_successors() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_BRANCH_PROB];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            3, // blocks
            0, 0, 4, 17, // block 0: return
            1, 0, 4, 18, // block 1: tail call
            2, 0, 4, 0, // block 2
            1, 2, 1, // return -> block 2 (malformed)
            1, 2, 1, // tail call -> block 2 (malformed)
            0, // block 2 has no successors
        ]);

        assert_eq!(decode(&bytes), Some(BTreeSet::from([0x100])));
    }

    #[test]
    fn rejects_duplicate_block_ids() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, 0];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks
            0, 0, 4, 0, // block 0: [0x100, 0x104)
            0, 4, 4, 0, // duplicate block 0
        ]);

        assert_eq!(decode(&bytes), None);
    }

    #[test]
    fn rejects_block_analysis_without_block_entries() {
        for analysis in [LLVM_BB_ADDR_MAP_BB_FREQ, LLVM_BB_ADDR_MAP_BRANCH_PROB] {
            let features = analysis | LLVM_BB_ADDR_MAP_OMIT_ENTRIES;
            assert_eq!(decode(&[LLVM_BB_ADDR_MAP_VERSION, features]), None);
        }
    }

    #[test]
    fn rejects_unknown_successor_ids() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_BRANCH_PROB];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            1, // blocks
            0, 0, 4, 16, // block 0: indirect branch
            1, 1, 1, // block 0 -> unknown block 1
        ]);

        assert_eq!(decode(&bytes), None);
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
