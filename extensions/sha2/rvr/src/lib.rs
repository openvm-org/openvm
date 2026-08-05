//! SHA-2 extension for rvr-openvm.
//!
//! Provides IR nodes for the SHA-256 and SHA-512 opcodes and the
//! `Sha2Extension` for lifting and executing them via double FFI.

use openvm_instructions::{
    riscv::{NUM_REGISTERS, REGISTER_BYTES},
    LocalOpcode,
};
use openvm_sha2_air::{Sha256Config, Sha2BlockHasherSubairConfig, Sha512Config};
use openvm_sha2_transpiler::Sha2Opcode;
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, FixedTraceRows, InstrAt, LiftedInstr, Variable,
};
use rvr_openvm_lift::{
    decode_variable, fixed_trace_rows_for_chip, max_main_memory_pages_for_contiguous_range,
    opcode_air_idx, AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction,
};

// SHA-512 has three independent ranges; its largest range is the 128-byte block.
const SHA2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    3 * max_main_memory_pages_for_contiguous_range(128);

fn decode_reg(value: u32) -> Variable {
    decode_variable(value, REGISTER_BYTES as u32, NUM_REGISTERS as u32)
}

const fn rows_to_u32(rows: usize) -> u32 {
    let rows_u32 = rows as u32;
    assert!(rows_u32 as usize == rows);
    rows_u32
}

const SHA256_ROWS_PER_BLOCK: u32 =
    rows_to_u32(<Sha256Config as Sha2BlockHasherSubairConfig>::ROWS_PER_BLOCK);
const SHA512_ROWS_PER_BLOCK: u32 =
    rows_to_u32(<Sha512Config as Sha2BlockHasherSubairConfig>::ROWS_PER_BLOCK);

/// IR node for a SHA-256 compress instruction.
///
/// Reads 32 bytes of state and 64 bytes of input block, applies SHA-256
/// compression, writes 32 bytes of new state to the destination pointer.
#[derive(Debug, Clone)]
pub struct Sha256Instr {
    /// Register index holding destination pointer (where new state is written).
    pub dst_ptr_reg: Variable,
    /// Register index holding state pointer (previous hash state).
    pub state_ptr_reg: Variable,
    /// Register index holding input pointer (message block).
    pub input_ptr_reg: Variable,
    /// AIR index of the SHA-256 block hasher chip (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: Option<AirIndex>,
}

impl ExtInstr for Sha256Instr {
    fn opname(&self) -> &str {
        "sha256"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_var(self.dst_ptr_reg);
        let st = ctx.read_var(self.state_ptr_reg);
        let inp = ctx.read_var(self.input_ptr_reg);
        ctx.emit_call("rvr_ext_sha256", &["state", &dst, &st, &inp]);
        // The FFI keeps pure and metered execution on one memcpy-based path. In compact
        // checkpoint mode its range wrappers mutate memory and dirty pages but deliberately emit
        // no per-access metadata, so preserve the remaining AIR clock schedule here. The four
        // post-compression words are the only values an independent replay chunk cannot derive
        // from its starting checkpoint.
        ctx.advance_timestamp(16);
        let state_bytes = Sha256Config::HASH_WORDS * Sha256Config::WORD_U8S;
        for offset in (0..state_bytes).step_by(size_of::<u64>()) {
            ctx.append_replay_value(&format!("peek_mem_u64(state, {dst} + {offset}ull)"));
        }
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        fixed_trace_rows_for_chip(self.block_hasher_chip_idx, SHA256_ROWS_PER_BLOCK)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// IR node for a SHA-512 compress instruction.
///
/// Reads 64 bytes of state and 128 bytes of input block, applies SHA-512
/// compression, writes 64 bytes of new state to the destination pointer.
#[derive(Debug, Clone)]
pub struct Sha512Instr {
    /// Register index holding destination pointer (where new state is written).
    pub dst_ptr_reg: Variable,
    /// Register index holding state pointer (previous hash state).
    pub state_ptr_reg: Variable,
    /// Register index holding input pointer (message block).
    pub input_ptr_reg: Variable,
    /// AIR index of the SHA-512 block hasher chip (ROWS_PER_BLOCK rows per instruction).
    pub block_hasher_chip_idx: Option<AirIndex>,
}

impl ExtInstr for Sha512Instr {
    fn opname(&self) -> &str {
        "sha512"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let dst = ctx.read_var(self.dst_ptr_reg);
        let st = ctx.read_var(self.state_ptr_reg);
        let inp = ctx.read_var(self.input_ptr_reg);
        ctx.emit_call("rvr_ext_sha512", &["state", &dst, &st, &inp]);
        ctx.advance_timestamp(32);
        let state_bytes = Sha512Config::HASH_WORDS * Sha512Config::WORD_U8S;
        for offset in (0..state_bytes).step_by(size_of::<u64>()) {
            ctx.append_replay_value(&format!("peek_mem_u64(state, {dst} + {offset}ull)"));
        }
    }

    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        fixed_trace_rows_for_chip(self.block_hasher_chip_idx, SHA512_ROWS_PER_BLOCK)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }
}

/// The SHA-2 extension (SHA-256 + SHA-512 opcodes).
/// Register this with the `ExtensionRegistry`.
pub struct Sha2Extension {
    sha256_block_hasher_chip_idx: Option<AirIndex>,
    sha512_block_hasher_chip_idx: Option<AirIndex>,
}

impl Sha2Extension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let sha256_main_chip_idx = opcode_air_idx(ctx, Sha2Opcode::SHA256)?;
        // The SHA-256 block hasher is registered adjacent to the main chip and
        // assigned the next AIR index (main_air_idx + 1) due to reverse registration order.
        let sha256_block_hasher_chip_idx = sha256_main_chip_idx.map(AirIndex::next);

        let sha512_main_chip_idx = opcode_air_idx(ctx, Sha2Opcode::SHA512)?;
        let sha512_block_hasher_chip_idx = sha512_main_chip_idx.map(AirIndex::next);

        Ok(Self {
            sha256_block_hasher_chip_idx,
            sha512_block_hasher_chip_idx,
        })
    }
}

impl RvrExtension for Sha2Extension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Sha2Opcode::SHA256.global_opcode_usize() {
            let dst_ptr_reg = decode_reg(insn.a);
            let state_ptr_reg = decode_reg(insn.b);
            let input_ptr_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(Sha256Instr {
                    dst_ptr_reg,
                    state_ptr_reg,
                    input_ptr_reg,
                    block_hasher_chip_idx: self.sha256_block_hasher_chip_idx,
                }),
                source_loc: None,
            }));
        }

        if opcode == Sha2Opcode::SHA512.global_opcode_usize() {
            let dst_ptr_reg = decode_reg(insn.a);
            let state_ptr_reg = decode_reg(insn.b);
            let input_ptr_reg = decode_reg(insn.c);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(Sha512Instr {
                    dst_ptr_reg,
                    state_ptr_reg,
                    input_ptr_reg,
                    block_hasher_chip_idx: self.sha512_block_hasher_chip_idx,
                }),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![("rvr_ext_sha2.h", include_str!("../c/rvr_ext_sha2.h"))]
    }

    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![(
            "librvr_openvm_ext_sha2_ffi.a",
            include_bytes!(env!("RVR_SHA2_FFI_STATICLIB")),
        )]
    }

    fn uses_memory_wrappers(&self) -> bool {
        true
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        SHA2_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

#[cfg(test)]
mod tests {
    use rvr_openvm_ir::{MemWidth, PageAddressSpace};

    use super::*;

    #[derive(Default)]
    struct TestEmitCtx {
        operations: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_var(&mut self, var: Variable) -> String {
            self.operations.push(format!("read(r{});", var.index()));
            format!("r{}", var.index())
        }

        fn peek_var(&mut self, _var: Variable) -> String {
            unreachable!()
        }

        fn advance_timestamp(&mut self, slots: u32) {
            self.operations.push(format!("advance({slots});"));
        }

        fn write_var(&mut self, _var: Variable, _val: &str) {
            unreachable!()
        }

        fn write_line(&mut self, _s: &str) {
            unreachable!()
        }

        fn emit_trap(&mut self) {
            unreachable!()
        }

        fn read_mem(&mut self, _base: &str, _offset: i16, _width: u8, _signed: bool) -> String {
            unreachable!()
        }

        fn write_mem(&mut self, _base: &str, _offset: i16, _val: &str, _width: u8) {
            unreachable!()
        }

        fn write_aligned_mem_block(&mut self, _addr: &str, _val: &str) {
            unreachable!()
        }

        fn reserve_preflight_timestamp_slots(&mut self, _slots: &str) {
            unreachable!()
        }

        fn append_replay_value(&mut self, value: &str) {
            self.operations.push(format!("replay_value({value});"));
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.operations
                .push(format!("{name}({});", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, _name: &str, _args: &[&str]) {
            unreachable!()
        }

        fn emit_call_expr(&mut self, _ret_ty: &str, _name: &str, _args: &[&str]) -> String {
            unreachable!()
        }

        fn emit_call_with_trace_result(
            &mut self,
            _ret_ty: &str,
            _name: &str,
            _args: &[&str],
        ) -> Option<String> {
            unreachable!()
        }

        fn trace_chip(&mut self, _chip_idx: u32, _count_expr: &str) {
            unreachable!()
        }

        fn trace_chip_if_nonzero(&mut self, _chip_idx: u32, _count_expr: &str) {
            unreachable!()
        }

        fn trace_page_access(
            &mut self,
            _addr: &str,
            _width: MemWidth,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }

        fn trace_page_access_u64_range(
            &mut self,
            _base_addr: &str,
            _num_dwords: &str,
            _addr_space: PageAddressSpace,
        ) {
            unreachable!()
        }
    }

    fn assert_checkpoint_shape(
        instr: &dyn ExtInstr,
        ffi_name: &str,
        remaining_slots: u32,
        expected_replay_values: usize,
    ) {
        assert!(instr.supports_preflight());
        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);
        assert_eq!(
            ctx.operations
                .iter()
                .filter(|operation| operation.starts_with("read("))
                .count(),
            3
        );
        assert_eq!(
            ctx.operations
                .iter()
                .filter(|operation| operation.starts_with(ffi_name))
                .count(),
            1
        );
        assert_eq!(
            ctx.operations
                .iter()
                .filter(|operation| operation.as_str() == format!("advance({remaining_slots});"))
                .count(),
            1
        );
        let replay_values = ctx
            .operations
            .iter()
            .filter(|operation| operation.starts_with("replay_value(peek_mem_u64("))
            .collect::<Vec<_>>();
        assert_eq!(replay_values.len(), expected_replay_values);
        for (index, operation) in replay_values.into_iter().enumerate() {
            assert!(operation.contains(&format!("+ {}ull", index * size_of::<u64>())));
        }
    }

    #[test]
    fn sha256_checkpoint_emits_exact_schedule_and_replay_values() {
        assert_checkpoint_shape(
            &Sha256Instr {
                dst_ptr_reg: Variable::new(1),
                state_ptr_reg: Variable::new(2),
                input_ptr_reg: Variable::new(3),
                block_hasher_chip_idx: None,
            },
            "rvr_ext_sha256(",
            16,
            4,
        );
    }

    #[test]
    fn sha512_checkpoint_emits_exact_schedule_and_replay_values() {
        assert_checkpoint_shape(
            &Sha512Instr {
                dst_ptr_reg: Variable::new(1),
                state_ptr_reg: Variable::new(2),
                input_ptr_reg: Variable::new(3),
                block_hasher_chip_idx: None,
            },
            "rvr_ext_sha512(",
            32,
            8,
        );
    }
}
