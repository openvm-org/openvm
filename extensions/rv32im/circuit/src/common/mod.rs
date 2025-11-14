#[cfg(feature = "aot")]
pub(crate) use aot::*;

#[cfg(feature = "aot")]
mod aot {
    use std::mem::offset_of;

    pub(crate) use openvm_circuit::arch::aot::common::*;
    use openvm_circuit::{
        arch::{
            execution_mode::{metered::memory_ctx::MemoryCtx, MeteredCtx},
            AotError, SystemConfig, VmExecState, ADDR_SPACE_OFFSET,
        },
        system::memory::{merkle::public_values::PUBLIC_VALUES_AS, online::GuestMemory, CHUNK},
    };
    use openvm_instructions::riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS};

    /// The minimum block size is 4, but RISC-V `lb` only requires alignment of 1 and `lh` only
    /// requires alignment of 2 because the instructions are implemented by doing an access of
    /// block size 4.
    const DEFAULT_U8_BLOCK_SIZE_BITS: u8 = 2;
    /// This is DIRTY because PAGE_BITS is a generic parameter of E2 context.
    const DEFAULT_PAGE_BITS: usize = 6;

    pub(crate) fn gpr_to_rv32_register(gpr: &str, rv32_reg: u8) -> String {
        let xmm_map_reg = rv32_reg / 2;
        if rv32_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 0\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 1\n")
        }
    }

    pub(crate) fn address_space_start_to_gpr(address_space: u32, gpr: &str) -> String {
        if address_space == 2 {
            if REG_AS2_PTR != gpr {
                return format!("    mov {gpr}, r15\n");
            }
            return "".to_string();
        }

        let xmm_map_reg = match address_space {
            1 => "xmm0",
            3 => "xmm1",
            4 => "xmm2",
            _ => unreachable!("Only address space 1, 2, 3, 4 is supported"),
        };
        format!("   pextrq {gpr}, {xmm_map_reg}, 1\n")
    }

    /*
    input:
    - riscv register number
    - gpr register to write into
    - is_gpr_force_write boolean

    output:
    - string representing the general purpose register that stores the value of register number `rv32_reg`
    - emitted assembly string that performs the move
    */
    pub(crate) fn xmm_to_gpr(
        rv32_reg: u8,
        gpr: &str,
        is_gpr_force_write: bool,
    ) -> (String, String) {
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] {
            // a/4 is overridden, b/4 is overridden
            if is_gpr_force_write {
                return (gpr.to_string(), format!("  mov {gpr}, {override_reg}\n"));
            }
            return (override_reg.to_string(), "".to_string());
        }
        let xmm_map_reg = rv32_reg / 2;
        if rv32_reg % 2 == 0 {
            (
                gpr.to_string(),
                format!("   pextrd {gpr}, xmm{xmm_map_reg}, 0\n"),
            )
        } else {
            (
                gpr.to_string(),
                format!("   pextrd {gpr}, xmm{xmm_map_reg}, 1\n"),
            )
        }
    }

    pub(crate) fn gpr_to_xmm(gpr: &str, rv32_reg: u8) -> String {
        if let Some(override_reg) = RISCV_TO_X86_OVERRIDE_MAP[rv32_reg as usize] {
            if gpr == override_reg {
                //already in correct location
                return "".to_string();
            }
            return format!("   mov {override_reg}, {gpr}\n");
        }
        let xmm_map_reg = rv32_reg / 2;
        if rv32_reg % 2 == 0 {
            format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 0\n")
        } else {
            format!("   pinsrd xmm{xmm_map_reg}, {gpr}, 1\n")
        }
    }
    pub(crate) fn update_adapter_heights_asm(
        config: &SystemConfig,
        _address_space: u32,
    ) -> Result<String, AotError> {
        let min_block_size_bits = config.memory_config.min_block_size_bits();
        if min_block_size_bits[RV32_REGISTER_AS as usize] != DEFAULT_U8_BLOCK_SIZE_BITS {
            println!("RV32_REGISTER_AS must have a minimum block size of 4");
            return Err(AotError::Other(String::from(
                "RV32_REGISTER_AS must have a minimum block size of 4",
            )));
        }
        if min_block_size_bits[RV32_MEMORY_AS as usize] != DEFAULT_U8_BLOCK_SIZE_BITS {
            println!("RV32_MEMORY_AS must have a minimum block size of 4");
            return Err(AotError::Other(String::from(
                "RV32_MEMORY_AS must have a minimum block size of 4",
            )));
        }
        if min_block_size_bits[PUBLIC_VALUES_AS as usize] != DEFAULT_U8_BLOCK_SIZE_BITS {
            println!("PUBLIC_VALUES_AS must have a minimum block size of 4");
            return Err(AotError::Other(String::from(
                "PUBLIC_VALUES_AS must have a minimum block size of 4",
            )));
        }

        // `update_adapter_heights_asm` rewrites the following code in ASM for
        // `on_memory_operation`: ```
        // pub fn update_adapter_heights_batch(
        //     &self,
        //     trace_heights: &mut [u32],
        //     address_space: u32,
        //     size_bits: u32,
        //     num: u32,
        // ) {
        //     let align_bits = unsafe {
        //         *self
        //             .min_block_size_bits
        //             .get_unchecked(address_space as usize)
        //     };
        //
        //     for adapter_bits in (align_bits as u32 + 1..=size_bits).rev() {
        //         let adapter_idx = self.adapter_offset + adapter_bits as usize - 1;
        //         debug_assert!(adapter_idx < trace_heights.len());
        //         unsafe {
        //             *trace_heights.get_unchecked_mut(adapter_idx) +=
        //                 num << (size_bits - adapter_bits + 1);
        //         }
        //     }
        // }
        // ```
        // 
        // For a specific RV32 instruction, the variables can be treated as constants at AOT
        // compilation time:
        // - `address_space`: always a constant because it is derived from an Instruction
        // - `num`: always 1 in `on_memory_operation`
        // - `align_bits`: always a constant because `address_space` is a constant
        // - `size_bits`: RV32 instruction always read 4 bytes(in the AIR level). So `size` is
        //   always 4 bytes. So `size_bits` is always 2.
        //
        // If we ignore the Native address space, `min_block_size_bits`` is always
        // `DEFAULT_U8_BLOCK_SIZE=4`. Therefore, `align_bits` is always 2. So the loop will
        // never be executed and we can leave the function empty.
        Ok("".to_string())
    }

    /// Generate ASM code for updating the boundary merkle heights.
    ///
    /// # Arguments
    ///
    /// * `config` - The system configuration.
    /// * `address_space` - The address space.
    /// * `pc` - The program counter of the current instruction.
    /// * `ptr_reg` - The register to store the accessed pointer. The caller should not expect the
    ///   value of this register to be preserved.
    /// * `reg1` - A register to store the intermediate result.
    /// * `reg2` - A register to store the intermediate result.
    ///
    /// # Returns
    ///
    /// The ASM code for updating the boundary merkle heights.
    pub(crate) fn update_boundary_merkle_heights_asm<F>(
        config: &SystemConfig,
        address_space: u32,
        pc: u32,
        ptr_reg: &str,
        reg1: &str,
        reg2: &str,
    ) -> Result<String, AotError> {
        // `update_boundary_merkle_heights_asm` rewrites the following code in ASM for
        // `on_memory_operation`: ```
        // pub fn label_to_index((addr_space, block_id): (u32, u32)) -> u64 {
        //     (((addr_space - ADDR_SPACE_OFFSET) as u64) << self.address_height) + block_id as u64
        // }
        //
        // pub(crate) fn update_boundary_merkle_heights(
        //     &mut self,
        //     address_space: u32,
        //     ptr: u32,
        //     size: u32,
        // ) {
        //     let num_blocks = (size + self.chunk - 1) >> self.chunk_bits;
        //     let start_chunk_id = ptr >> self.chunk_bits;
        //     let start_block_id = if self.chunk == 1 {
        //         start_chunk_id
        //     } else {
        //         self.memory_dimensions
        //             .label_to_index((address_space, start_chunk_id)) as u32
        //     };
        //     // Because `self.chunk == 1 << self.chunk_bits`
        //     let end_block_id = start_block_id + num_blocks;
        //     let start_page_id = start_block_id >> PAGE_BITS;
        //     let end_page_id = ((end_block_id - 1) >> PAGE_BITS) + 1;

        //     for page_id in start_page_id..end_page_id {
        //         if self.page_indices.insert(page_id as usize) {
        //             self.page_access_count += 1;
        //             // SAFETY: address_space passed is usually a hardcoded constant or derived
        // from an             // Instruction where it is bounds checked before passing
        //             unsafe {
        //                 *self
        //                     .addr_space_access_count
        //                     .get_unchecked_mut(address_space as usize) += 1;
        //             }
        //         }
        //     }
        // }
        // ```
        // 
        // For a specific RV32 instruction, the variables can be treated as constants at AOT compilation time:
        // Inputs:
        // - `chunk`: always 8(CHUNK) because we only support when continuation is enabled.
        // - `address_space`: always a constant because it is derived from an Instruction
        // - `size`: RV32 instruction always read 4 bytes(in the AIR level).
        // - `self.memory_dimensions.address_height`: known at AOT compilation time because it is derived from the memory configuration.
        // Inside the function body:
        // - `num_blocks`: `(size + self.chunk - 1) >> self.chunk_bits = (4 + 8 - 1) >> 3 = 1`
        // - `as_offset = (addr_space - ADDR_SPACE_OFFSET) as u64) << self.address_height)`: constant because `address_space` and `address_height` constant
        // - `start_chunk_id`: `ptr >> self.chunk_bits`
        // - `start_block_id`: `start_chunk_id + as_offset`
        // - `end_block_id`: `start_block_id + num_blocks = start_block_id +1`
        // - `start_page_id`: `start_block_id >> PAGE_BITS`
        // - `end_page_id`: ((end_block_id - 1) >> PAGE_BITS) + 1 = start_block_id >> PAGE_BITS + 1;
        //
        // Therefore the loop only iterates once for `page_id = start_page_id`.

        let initial_block_size: usize = config.initial_block_size();
        if initial_block_size != CHUNK {
            return Err(AotError::Other(format!(
                "initial_block_size must be {CHUNK}, got {initial_block_size}"
            )));
        }
        let chunk_bits = CHUNK.ilog2();
        let as_offset = ((address_space - ADDR_SPACE_OFFSET) as u64)
            << (config.memory_config.memory_dimensions().address_height);

        let mut asm_str = String::new();
        // `start_chunk_id`: `ptr >> self.chunk_bits`
        asm_str += &format!("    shr {ptr_reg}, {chunk_bits}\n");
        // `start_block_id`: `start_chunk_id + as_offset`
        asm_str += &format!("    add {ptr_reg}, {as_offset}\n");
        // `start_page_id`: `start_block_id >> PAGE_BITS`
        // NOTE: This is DIRTY because PAGE_BITS is a generic parameter of E2 context.
        asm_str += &format!("    shr {ptr_reg}, {DEFAULT_PAGE_BITS}\n");

        let memory_ctx_offset = offset_of!(VmExecState<F, GuestMemory, MeteredCtx>, ctx)
            + offset_of!(MeteredCtx, memory_ctx);
        let page_indices_ptr_offset =
            memory_ctx_offset + offset_of!(MemoryCtx<DEFAULT_PAGE_BITS>, page_indices);
        let page_access_count_offset =
            memory_ctx_offset + offset_of!(MemoryCtx<DEFAULT_PAGE_BITS>, page_access_count);
        let addr_space_access_count_ptr_offset =
            memory_ctx_offset + offset_of!(MemoryCtx<DEFAULT_PAGE_BITS>, addr_space_access_count);
        let inserted_label = format!(".asm_execute_pc_{pc}_inserted");
        // The next section is the implementation of `BitSet::insert` in ASM.
        // pub fn insert(&mut self, index: usize) -> bool {
        //     let word_index = index >> 6;
        //     let bit_index = index & 63;
        //     let mask = 1u64 << bit_index;
        //     let word = unsafe { self.words.get_unchecked_mut(word_index) };
        //     let was_set = (*word & mask) != 0;
        //     *word |= mask;
        //     !was_set
        // }

        // Start with `ptr_reg = index`
        // `reg1 = word_index`
        asm_str += &format!("    mov {reg1}, {ptr_reg}\n");
        asm_str += &format!("    shr {reg1}, 6\n");
        // `ptr_reg = bit_index = index & 63`
        asm_str += &format!("    and {ptr_reg}, 63\n");
        // `reg2 = mask = 1u64 << bit_index`
        asm_str += &format!("    mov {reg2}, 1\n");
        asm_str += &format!("    shlx {reg2}, {reg2}, {ptr_reg}\n");
        // `ptr_reg = self.page_indices.ptr`
        asm_str +=
            &format!("    mov {ptr_reg}, [{REG_EXEC_STATE_PTR} + {page_indices_ptr_offset}]\n");

        // `reg1 = word_ptr = &self.words.get_unchecked_mut(word_index)`
        asm_str += &format!("    lea {reg1}, [{ptr_reg} + {reg1} * 8]\n");
        // `ptr_reg = word = *word_ptr`
        asm_str += &format!("    mov {ptr_reg}, [{reg1}]\n");

        // `test (*word & mask)`
        asm_str += &format!("    test {ptr_reg}, {reg2}\n");
        asm_str += &format!("    jnz {inserted_label}\n");
        // When (*word & mask) == 0
        // `*word += mask`
        asm_str += &format!("    add {ptr_reg}, {reg2}\n");
        asm_str += &format!("    mov [{reg1}], {ptr_reg}\n");
        // reg1 = &self.page_access_count`
        asm_str +=
            &format!("    lea {reg1}, [{REG_EXEC_STATE_PTR} + {page_access_count_offset}]\n");
        // self.page_access_count += 1;
        asm_str += &format!("    add dword ptr [{reg1}], 1\n");
        // reg1 = &addr_space_access_count.as_ptr()
        asm_str += &format!(
            "    lea {reg1}, [{REG_EXEC_STATE_PTR} + {addr_space_access_count_ptr_offset}]\n"
        );
        asm_str += &format!("    mov {reg1}, [{reg1}]\n");
        // self.addr_space_access_count[address_space] += 1;
        asm_str += &format!("    add dword ptr [{reg1} + {address_space} * 4], 1\n");
        asm_str += &format!("{inserted_label}:\n");
        // Inserted, do nothing

        Ok(asm_str)
    }

    /// Assumption: `REG_TRACE_HEIGHT` is the pointer of `trace_heights``.
    pub(crate) fn update_height_change_asm(
        chip_idx: usize,
        height_delta: u32,
    ) -> Result<String, AotError> {
        let mut asm_str = String::new();
        // `update_height_change_asm` rewrites the following code in ASM for `on_height_change`:
        // ```
        // pub fn on_height_change(&mut self, chip_idx: usize, height_delta: u32) {
        //     self.trace_heights[chip_idx] += height_delta;
        // }
        // ```
        asm_str +=
            &format!("    add dword ptr [{REG_TRACE_HEIGHT} + {chip_idx} * 4], {height_delta}\n");
        Ok(asm_str)
    }
}
