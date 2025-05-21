// Initial version taken from https://github.com/succinctlabs/sp1/blob/v2.0.0/crates/core/executor/src/disassembler/elf.rs under MIT License
// and https://github.com/risc0/risc0/blob/f61379bf69b24d56e49d6af96a3b284961dcc498/risc0/binfmt/src/elf.rs#L34 under Apache License
use std::{cmp::min, collections::BTreeMap, fmt::Debug};
#[cfg(feature = "function-span")]
use std::{
    collections::{hash_map::Entry, HashMap},
    io::Write,
};

use elf::{
    abi::{EM_RISCV, ET_EXEC, PF_X, PT_LOAD},
    endian::LittleEndian,
    file::Class,
    ElfBytes,
};
use eyre::{self, bail, ContextCompat};
#[cfg(feature = "function-span")]
use openvm_instructions::exe::FnBound;
use openvm_instructions::{exe::FnBounds, program::MAX_ALLOWED_PC};
use openvm_platform::WORD_SIZE;

/// RISC-V 32IM ELF (Executable and Linkable Format) File.
///
/// This file represents a binary in the ELF format, specifically the RISC-V 32IM architecture
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
}

impl Elf {
    /// Create a new [Elf].
    pub(crate) const fn new(
        instructions: Vec<u32>,
        pc_start: u32,
        pc_base: u32,
        memory_image: BTreeMap<u32, u32>,
        fn_bounds: FnBounds,
    ) -> Self {
        Self {
            instructions,
            pc_start,
            pc_base,
            memory_image,
            fn_bounds,
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

        // Some sanity checks to make sure that the ELF file is valid.
        if elf.ehdr.class != Class::ELF32 {
            bail!("Not a 32-bit ELF");
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
                                end: (symbol.st_value + symbol.st_size - (WORD_SIZE as u64)) as u32,
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
        if entry >= max_mem || entry % WORD_SIZE as u32 != 0 {
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

        // Only read segments that are executable instructions that are also PT_LOAD.
        for segment in segments.iter().filter(|x| x.p_type == PT_LOAD) {
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
            if vaddr % WORD_SIZE as u32 != 0 {
                bail!("vaddr {vaddr:08x} is unaligned");
            }

            // If the virtual address is less than the first memory address, then update the first
            // memory address.
            if (segment.p_flags & PF_X) != 0 && base_address > vaddr {
                base_address = vaddr;
            }

            // Get the offset to the segment.
            let offset: u32 = segment.p_offset.try_into()?;

            // Read the segment and decode each word as an instruction.
            for i in (0..mem_size).step_by(WORD_SIZE) {
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
                let len = min(file_size - i, WORD_SIZE as u32);
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

        Ok(Elf::new(
            instructions,
            entry,
            base_address,
            image,
            fn_bounds,
        ))
    }
}
