// Initial version taken from https://github.com/succinctlabs/sp1/blob/v2.0.0/crates/core/executor/src/disassembler/elf.rs under MIT License
// and https://github.com/risc0/risc0/blob/f61379bf69b24d56e49d6af96a3b284961dcc498/risc0/binfmt/src/elf.rs#L34 under Apache License
use std::{
    cmp::min,
    collections::{BTreeMap, HashMap},
    fmt::Debug,
};

#[cfg(feature = "function-span")]
use axvm_instructions::exe::FnBound;
use axvm_instructions::exe::FnBounds;
use axvm_platform::WORD_SIZE;
use elf::{
    abi::{EM_RISCV, ET_EXEC, PF_X, PT_LOAD},
    endian::LittleEndian,
    file::Class,
    ElfBytes,
};
use eyre::{self, bail, ContextCompat};
use num_bigint_dig::BigUint;

pub const ELF_DEFAULT_MAX_NUM_PUBLIC_VALUES: usize = 32;

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
    pub(crate) instructions: Vec<u32>,
    /// The start address of the program.
    pub(crate) pc_start: u32,
    /// The base address of the program.
    pub(crate) pc_base: u32,
    /// The initial memory image, useful for global constants.
    pub(crate) memory_image: BTreeMap<u32, u32>,
    /// The upper bound of the number of public values the program would publish.
    /// TODO: read from project config.
    pub(crate) max_num_public_values: usize,
    /// Field arithmetic configuration.
    pub(crate) supported_moduli: Vec<String>,
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
        supported_moduli: Vec<String>,
        fn_bounds: FnBounds,
    ) -> Self {
        Self {
            instructions,
            pc_start,
            pc_base,
            memory_image,
            max_num_public_values: ELF_DEFAULT_MAX_NUM_PUBLIC_VALUES,
            supported_moduli,
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
                for symbol in symtab.iter() {
                    if symbol.st_symtype() == elf::abi::STT_FUNC {
                        fn_bounds.insert(
                            symbol.st_value as u32,
                            FnBound {
                                start: symbol.st_value as u32,
                                end: (symbol.st_value + symbol.st_size - (WORD_SIZE as u64)) as u32,
                                name: stringtab.get(symbol.st_name as usize).unwrap().to_string(),
                            },
                        );
                    }
                }
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

        // Get the prime moduli from the .axiom section.
        let axiom_section_header = elf
            .section_header_by_name(".axiom")
            .expect("section table should be parsable");
        let mut supported_moduli = HashMap::new();
        if let Some(shdr) = axiom_section_header {
            let data = elf.section_data(&shdr).unwrap();
            let ptr = std::cell::Cell::new(0);
            let next = || {
                let result = *data
                    .0
                    .get(ptr.get())
                    .expect("unexpected end of .axiom section");
                ptr.set(ptr.get() + 1);
                result
            };
            let next_u32 = || u32::from_le_bytes(std::array::from_fn(|_| next()));

            while ptr.get() < data.0.len() {
                match next() {
                    1 => {
                        let mod_idx = next() as u32;
                        eprintln!("discovered modulus #{mod_idx}");
                        supported_moduli.insert(mod_idx, {
                            let cnt_bytes = next_u32() as usize;
                            BigUint::from_bytes_le(
                                &(0..cnt_bytes).map(|_| next()).collect::<Vec<_>>(),
                            )
                            .to_str_radix(10)
                        });
                    }
                    _ => {
                        unimplemented!()
                    }
                }
            }
        } else {
            eprintln!("no .axiom section found");
        }

        Ok(Elf::new(
            instructions,
            entry,
            base_address,
            image,
            (0..supported_moduli.len())
                .map(|x| supported_moduli[&(x as u32)].clone())
                .collect(),
            fn_bounds,
        ))
    }
}
