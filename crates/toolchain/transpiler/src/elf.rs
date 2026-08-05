// Initial version taken from https://github.com/succinctlabs/sp1/blob/v2.0.0/crates/core/executor/src/disassembler/elf.rs under MIT License
// and https://github.com/risc0/risc0/blob/f61379bf69b24d56e49d6af96a3b284961dcc498/risc0/binfmt/src/elf.rs#L34 under Apache License
use std::{
    cmp::min,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
    mem::size_of,
};
#[cfg(feature = "function-span")]
use std::{
    collections::{hash_map::Entry, HashMap},
    io::Write,
};

use elf::{
    abi::{EM_RISCV, ET_EXEC, PF_X, PT_LOAD, SHN_UNDEF, STT_FUNC},
    endian::LittleEndian,
    file::Class,
    ElfBytes,
};
use eyre::{self, bail, ContextCompat};
#[cfg(feature = "function-span")]
use openvm_instructions::exe::FnBound;
use openvm_instructions::{exe::FnBounds, program::MAX_ALLOWED_PC};

/// The size of a RISC-V instruction in bytes.
const ELF_WORD_SIZE: usize = 4;
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
fn llvm_block_starts(mut bytes: &[u8]) -> Option<BTreeSet<u32>> {
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

fn llvm_block_starts_from_elf(elf: &ElfBytes<'_, LittleEndian>) -> BTreeSet<u32> {
    let mut block_starts = BTreeSet::new();
    if let Some(sections) = elf.section_headers() {
        for section in sections
            .iter()
            .filter(|section| section.sh_type == SHT_LLVM_BB_ADDR_MAP)
        {
            let Ok((bytes, None)) = elf.section_data(&section) else {
                continue;
            };
            if let Some(starts) = llvm_block_starts(bytes) {
                block_starts.extend(starts);
            }
        }
    }
    block_starts
}

/// Uses aligned, defined function entries as conservative block starts.
fn function_symbol_block_starts(elf: &ElfBytes<'_, LittleEndian>) -> BTreeSet<u32> {
    elf.symbol_table()
        .ok()
        .flatten()
        .map(|(symbols, _)| {
            symbols
                .iter()
                .filter(|symbol| symbol.st_symtype() == STT_FUNC && symbol.st_shndx != SHN_UNDEF)
                .filter_map(|symbol| u32::try_from(symbol.st_value).ok())
                .filter(|pc| pc.is_multiple_of(ELF_WORD_SIZE as u32))
                .collect()
        })
        .unwrap_or_default()
}

/// Combines LLVM's machine-block map with function entries from the final ELF.
fn cfg_block_starts(elf: &ElfBytes<'_, LittleEndian>) -> BTreeSet<u32> {
    let mut block_starts = function_symbol_block_starts(elf);
    block_starts.extend(llvm_block_starts_from_elf(elf));
    block_starts
}

/// RISC-V 64IM ELF (Executable and Linkable Format) File.
///
/// This file represents a binary in the ELF format, specifically the RISC-V 64IM architecture
/// with the following extensions:
///
/// - Base Integer Instruction Set (I)
/// - Integer Multiplication and Division (M)
///
/// This format is commonly used in embedded systems and is supported by many compilers.
#[derive(Debug, Clone)]
pub struct Elf {
    /// The instructions of the program encoded as 32-bits.
    pub instructions: Vec<u32>,
    /// The start address of the program.
    pub(crate) pc_start: u32,
    /// The base address of the program.
    pub(crate) pc_base: u32,
    /// The initial memory image, useful for global constants.
    pub(crate) memory_image: BTreeMap<u32, u32>,
    /// Debug info for spanning benchmark metrics by function.
    pub(crate) fn_bounds: FnBounds,
    /// Machine basic-block PCs retained from the final ELF.
    pub(crate) cfg_block_starts: BTreeSet<u32>,
}

impl Elf {
    /// Create a new [Elf].
    pub(crate) const fn new(
        instructions: Vec<u32>,
        pc_start: u32,
        pc_base: u32,
        memory_image: BTreeMap<u32, u32>,
        fn_bounds: FnBounds,
        cfg_block_starts: BTreeSet<u32>,
    ) -> Self {
        Self {
            instructions,
            pc_start,
            pc_base,
            memory_image,
            fn_bounds,
            cfg_block_starts,
        }
    }

    /// Parse the ELF file into a vector of 32-bit encoded instructions and the first memory
    /// address.
    ///
    /// # Errors
    ///
    /// This function may return an error if the ELF is not valid.
    ///
    /// Reference: [Executable and Linkable Format](https://en.wikipedia.org/wiki/Executable_and_Linkable_Format)
    pub fn decode(input: &[u8], max_mem: u32) -> eyre::Result<Self> {
        let mut image: BTreeMap<u32, u32> = BTreeMap::new();

        // Parse the ELF file assuming that it is little-endian..
        let elf = ElfBytes::<LittleEndian>::minimal_parse(input)
            .map_err(|err| eyre::eyre!("Elf parse error: {err}"))?;

        let mut cfg_block_starts = cfg_block_starts(&elf);

        // Some sanity checks to make sure that the ELF file is valid.
        if elf.ehdr.class != Class::ELF64 {
            bail!("Not a 64-bit ELF");
        } else if elf.ehdr.e_machine != EM_RISCV {
            bail!("Invalid machine type, must be RISC-V");
        } else if elf.ehdr.e_type != ET_EXEC {
            bail!("Invalid ELF type, must be executable");
        }

        #[cfg(not(feature = "function-span"))]
        let fn_bounds = Default::default();

        #[cfg(feature = "function-span")]
        let mut fn_bounds = FnBounds::new();
        #[cfg(feature = "function-span")]
        {
            if let Some((symtab, stringtab)) = elf.symbol_table()? {
                let mut fn_names = Vec::new();
                for symbol in symtab.iter() {
                    if symbol.st_symtype() == elf::abi::STT_FUNC {
                        let raw_name = stringtab.get(symbol.st_name as usize).unwrap().to_string();
                        let demangled_name = rustc_demangle::demangle(&raw_name).to_string();
                        fn_names.push((demangled_name, symbol.st_name));
                    }
                }

                let mut buf = Vec::new();
                let mut offsets = HashMap::new();
                buf.push(0);
                for (name, st_name) in fn_names {
                    if let Entry::Vacant(e) = offsets.entry(st_name) {
                        let offset = buf.len();
                        e.insert(offset);
                        buf.extend_from_slice(name.as_bytes());
                        buf.push(0);
                    }
                }

                for symbol in symtab.iter() {
                    if symbol.st_symtype() == elf::abi::STT_FUNC {
                        fn_bounds.insert(
                            symbol.st_value as u32,
                            FnBound {
                                start: symbol.st_value as u32,
                                end: (symbol.st_value + symbol.st_size - (ELF_WORD_SIZE as u64))
                                    as u32,
                                name: offsets[&symbol.st_name].to_string(),
                            },
                        );
                    }
                }

                let guest_symbols_path = std::env::var("GUEST_SYMBOLS_PATH")
                    .map_err(|e| eyre::eyre!("{e}: GUEST_SYMBOLS_PATH"))?;
                let mut guest_symbols_file =
                    std::fs::File::create(&guest_symbols_path).map_err(|e| {
                        eyre::eyre!(
                            "Failed to create guest symbols file at {guest_symbols_path}: {e}"
                        )
                    })?;
                guest_symbols_file.write_all(buf.as_slice())?;
            } else {
                println!("No symbol table found");
            }
        }

        // Get the entrypoint of the ELF file as an u32.
        let entry: u32 = elf
            .ehdr
            .e_entry
            .try_into()
            .map_err(|err| eyre::eyre!("e_entry was larger than 32 bits. {err}"))?;

        // Make sure the entrypoint is valid.
        if entry >= max_mem || !entry.is_multiple_of(ELF_WORD_SIZE as u32) {
            bail!("Invalid entrypoint");
        }

        // Get the segments of the ELF file.
        let segments = elf
            .segments()
            .ok_or_else(|| eyre::eyre!("Missing segment table"))?;
        if segments.len() > 256 {
            bail!("Too many program headers");
        }

        let mut instructions: Vec<u32> = Vec::new();
        let mut base_address = u32::MAX;
        // Track the end of the last executable segment to detect non-contiguous executable
        // segments.
        let mut last_exec_end: Option<u32> = None;

        // Collect and sort PT_LOAD segments by virtual address to ensure executable
        // segment contiguity checks are correct regardless of ELF header ordering.
        let mut load_segments: Vec<_> = segments.iter().filter(|x| x.p_type == PT_LOAD).collect();
        load_segments.sort_by_key(|s| s.p_vaddr);

        for segment in load_segments {
            // Get the file size of the segment as an u32.
            let file_size: u32 = segment.p_filesz.try_into()?;
            if file_size >= max_mem {
                bail!("invalid segment file_size");
            }

            // Get the memory size of the segment as an u32.
            let mem_size: u32 = segment.p_memsz.try_into()?;
            if mem_size >= max_mem {
                bail!("Invalid segment mem_size");
            }

            // Get the virtual address of the segment as an u32.
            let vaddr: u32 = segment.p_vaddr.try_into()?;
            if !vaddr.is_multiple_of(ELF_WORD_SIZE as u32) {
                bail!("vaddr {vaddr:08x} is unaligned");
            }

            // Track executable segments and reject non-contiguous ones.
            if (segment.p_flags & PF_X) != 0 {
                if let Some(prev_end) = last_exec_end {
                    if vaddr != prev_end {
                        bail!(
                            "Non-contiguous executable segments are not supported: \
                             previous segment ended at 0x{prev_end:08x}, \
                             next segment starts at 0x{vaddr:08x}"
                        );
                    }
                }
                if base_address > vaddr {
                    base_address = vaddr;
                }
                last_exec_end = Some(
                    vaddr
                        .checked_add(mem_size)
                        .ok_or_else(|| eyre::eyre!("executable segment end address overflow"))?,
                );
            }

            // Get the offset to the segment.
            let offset: u32 = segment.p_offset.try_into()?;

            // Read the segment and decode each word as an instruction.
            for i in (0..mem_size).step_by(ELF_WORD_SIZE) {
                let addr = vaddr
                    .checked_add(i)
                    .ok_or_else(|| eyre::eyre!("vaddr overflow"))?;
                if addr >= max_mem {
                    bail!(
                        "address [0x{addr:08x}] exceeds maximum address for guest programs [0x{max_mem:08x}]"
                    );
                } else if addr > MAX_ALLOWED_PC && (segment.p_flags & PF_X) != 0 {
                    bail!("instruction address [0x{addr:08x}] exceeds maximum PC [0x{MAX_ALLOWED_PC:08x}]");
                }

                // If we are reading past the end of the file, then break.
                if i >= file_size {
                    image.insert(addr, 0);
                    continue;
                }

                // Get the word as an u32 but make sure we don't read pass the end of the file.
                let mut word = 0;
                let len = min(file_size - i, ELF_WORD_SIZE as u32);
                for j in 0..len {
                    let offset = (offset + i + j) as usize;
                    let byte = input.get(offset).context("Invalid segment offset")?;
                    word |= u32::from(*byte) << (j * 8);
                }
                image.insert(addr, word);
                if (segment.p_flags & PF_X) != 0 {
                    instructions.push(word);
                }
            }
        }

        let instruction_bytes = u32::try_from(instructions.len())?
            .checked_mul(ELF_WORD_SIZE as u32)
            .context("instruction byte length overflow")?;
        let program_end = base_address
            .checked_add(instruction_bytes)
            .context("program end address overflow")?;
        cfg_block_starts.retain(|pc| {
            pc.is_multiple_of(ELF_WORD_SIZE as u32) && (base_address..program_end).contains(pc)
        });

        Ok(Elf::new(
            instructions,
            entry,
            base_address,
            image,
            fn_bounds,
            cfg_block_starts,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_llvm_basic_block_address_map() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_CALLSITES];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[
            2, // blocks
            0, 0, 0, 4, 8, // block 0: [0x100, 0x104)
            1, 4, 1, 4, 4, 1, // block 1: [0x108, 0x110), one callsite at 0x10c
        ]);

        assert_eq!(
            llvm_block_starts(&bytes),
            Some(BTreeSet::from([0x100, 0x108]))
        );
    }

    #[test]
    fn rejects_truncated_llvm_basic_block_address_map() {
        assert_eq!(llvm_block_starts(&[LLVM_BB_ADDR_MAP_VERSION]), None);
    }

    #[test]
    fn omitted_entries_do_not_iterate_declared_block_count() {
        let mut bytes = vec![LLVM_BB_ADDR_MAP_VERSION, LLVM_BB_ADDR_MAP_OMIT_ENTRIES];
        bytes.extend_from_slice(&0x100u64.to_le_bytes());
        bytes.extend_from_slice(&[0xff, 0xff, 0xff, 0xff, 0x0f]);

        assert_eq!(llvm_block_starts(&bytes), Some(BTreeSet::from([0x100])));
    }

    #[test]
    fn falls_back_to_function_symbols_without_llvm_map() {
        let bytes = include_bytes!("../../../sdk/programs/examples/fibonacci.elf");
        let elf = ElfBytes::<LittleEndian>::minimal_parse(bytes).unwrap();

        assert!(elf
            .section_headers()
            .unwrap()
            .iter()
            .all(|section| section.sh_type != SHT_LLVM_BB_ADDR_MAP));
        assert_eq!(cfg_block_starts(&elf), function_symbol_block_starts(&elf));
        assert!(!cfg_block_starts(&elf).is_empty());
    }
}
