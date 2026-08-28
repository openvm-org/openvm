// Initial version taken from https://github.com/succinctlabs/sp1/blob/v2.0.0/crates/core/executor/src/disassembler/elf.rs under MIT License
// and https://github.com/risc0/risc0/blob/f61379bf69b24d56e49d6af96a3b284961dcc498/risc0/binfmt/src/elf.rs#L34 under Apache License
mod llvm_bb_addr_map;

use std::{
    cmp::min,
    collections::{BTreeMap, BTreeSet},
    fmt::Debug,
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
    block_starts.extend(llvm_bb_addr_map::block_starts(elf));
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
    pub fn decode(input: &[u8], max_mem: u64) -> eyre::Result<Self> {
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
        if u64::from(entry) >= max_mem || !entry.is_multiple_of(ELF_WORD_SIZE as u32) {
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
        let mut last_exec_end: Option<u64> = None;

        // Collect and sort PT_LOAD segments by virtual address to ensure executable
        // segment contiguity checks are correct regardless of ELF header ordering.
        let mut load_segments: Vec<_> = segments.iter().filter(|x| x.p_type == PT_LOAD).collect();
        load_segments.sort_by_key(|s| s.p_vaddr);

        for segment in load_segments {
            // Get the file size of the segment as an u32.
            let file_size: u32 = segment.p_filesz.try_into()?;
            if u64::from(file_size) >= max_mem {
                bail!("invalid segment file_size");
            }

            // Get the memory size of the segment as an u32.
            let mem_size: u32 = segment.p_memsz.try_into()?;
            if u64::from(mem_size) >= max_mem {
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
                    if u64::from(vaddr) != prev_end {
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
                    u64::from(vaddr)
                        .checked_add(u64::from(mem_size))
                        .ok_or_else(|| eyre::eyre!("executable segment end address overflow"))?,
                );
            }

            // Get the offset to the segment.
            let offset: u32 = segment.p_offset.try_into()?;

            // Read the segment and decode each word as an instruction.
            for i in (0..mem_size).step_by(ELF_WORD_SIZE) {
                let addr_u64 = u64::from(vaddr)
                    .checked_add(u64::from(i))
                    .ok_or_else(|| eyre::eyre!("vaddr overflow"))?;
                if addr_u64 >= max_mem {
                    bail!(
                        "address [0x{addr_u64:08x}] exceeds maximum address for guest programs [0x{max_mem:08x}]"
                    );
                }
                let addr = u32::try_from(addr_u64)
                    .map_err(|_| eyre::eyre!("address exceeds the u32 guest address space"))?;
                if addr > MAX_ALLOWED_PC && (segment.p_flags & PF_X) != 0 {
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

        let instruction_bytes = u64::try_from(instructions.len())?
            .checked_mul(ELF_WORD_SIZE as u64)
            .context("instruction byte length overflow")?;
        let program_end = u64::from(base_address)
            .checked_add(instruction_bytes)
            .context("program end address overflow")?;
        cfg_block_starts.retain(|pc| {
            pc.is_multiple_of(ELF_WORD_SIZE as u32)
                && (u64::from(base_address)..program_end).contains(&u64::from(*pc))
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

    fn put_u16(bytes: &mut [u8], offset: usize, value: u16) {
        bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
    }

    fn single_instruction_elf(vaddr: u32, instruction: u32) -> Vec<u8> {
        const ELF_HEADER_SIZE: usize = 64;
        const PROGRAM_HEADER_SIZE: u16 = 56;
        const SEGMENT_OFFSET: usize = 0x1000;

        let mut bytes = vec![0; SEGMENT_OFFSET + ELF_WORD_SIZE];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2; // ELFCLASS64
        bytes[5] = 1; // ELFDATA2LSB
        bytes[6] = 1; // EV_CURRENT
        put_u16(&mut bytes, 16, ET_EXEC);
        put_u16(&mut bytes, 18, EM_RISCV);
        put_u32(&mut bytes, 20, 1);
        put_u64(&mut bytes, 24, u64::from(vaddr));
        put_u64(&mut bytes, 32, ELF_HEADER_SIZE as u64);
        put_u16(&mut bytes, 52, ELF_HEADER_SIZE as u16);
        put_u16(&mut bytes, 54, PROGRAM_HEADER_SIZE);
        put_u16(&mut bytes, 56, 1);

        let ph = ELF_HEADER_SIZE;
        put_u32(&mut bytes, ph, PT_LOAD);
        put_u32(&mut bytes, ph + 4, elf::abi::PF_R | PF_X);
        put_u64(&mut bytes, ph + 8, SEGMENT_OFFSET as u64);
        put_u64(&mut bytes, ph + 16, u64::from(vaddr));
        put_u64(&mut bytes, ph + 24, u64::from(vaddr));
        put_u64(&mut bytes, ph + 32, ELF_WORD_SIZE as u64);
        put_u64(&mut bytes, ph + 40, ELF_WORD_SIZE as u64);
        put_u64(&mut bytes, ph + 48, ELF_WORD_SIZE as u64);
        put_u32(&mut bytes, SEGMENT_OFFSET, instruction);
        bytes
    }

    #[test]
    fn falls_back_to_function_symbols_without_llvm_map() {
        let bytes = include_bytes!("../../../sdk/programs/examples/fibonacci.elf");
        let elf = ElfBytes::<LittleEndian>::minimal_parse(bytes).unwrap();

        assert!(llvm_bb_addr_map::block_starts(&elf).is_empty());
        assert_eq!(cfg_block_starts(&elf), function_symbol_block_starts(&elf));
        assert!(!cfg_block_starts(&elf).is_empty());
    }

    #[test]
    fn decodes_instruction_at_maximum_pc() {
        let instruction = 0x0000_0013;
        let bytes = single_instruction_elf(MAX_ALLOWED_PC, instruction);

        let elf = Elf::decode(&bytes, 1u64 << 32).unwrap();

        assert_eq!(elf.pc_base, MAX_ALLOWED_PC);
        assert_eq!(elf.pc_start, MAX_ALLOWED_PC);
        assert_eq!(elf.instructions, vec![instruction]);
    }
}
