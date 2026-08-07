//! RV64 IO instruction lifting and host callbacks.

use std::{ffi::c_void, ops::Range};

use openvm_circuit::arch::rvr::io::{checked_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{
        is_valid_register_pointer, MEMORY_AS, REGISTER_AS, REGISTER_BYTES, REGISTER_NUM_LIMBS,
    },
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_platform::WORD_SIZE;
use openvm_riscv_transpiler::{HintStoreOpcode, RevealOpcode, MAX_HINT_BUFFER_DWORDS};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, MemWidth, PageAddressSpace,
};
use rvr_openvm_lift::{
    air_index_to_c, max_main_memory_pages_for_contiguous_range, opcode_air_idx, AirIndex,
    ExtensionError, RvrExtension, RvrExtensionCtx, RvrRuntimeExtension,
};

use crate::instruction::{decode_imm_cg, decode_reg, Reg};

// HINT_BUFFER writes the maximum hint payload as one contiguous range.
const IO_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
    max_main_memory_pages_for_contiguous_range(MAX_HINT_BUFFER_DWORDS * WORD_SIZE);

/// HINT_STORED: pop one register word (8 bytes) from the hint stream into `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub(crate) struct HintStoreWInstr {
    pub(crate) ptr_reg: Reg,
}

impl ExtInstr for HintStoreWInstr {
    fn opname(&self) -> &str {
        "hint_storew"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_var(self.ptr_reg);
        if !ctx.is_preflight() {
            ctx.count_fixed_replay_values(1);
            ctx.emit_checked_call_without_page_flush("openvm_hint_storew", &[&ptr]);
            ctx.trace_page_access(
                &ptr,
                MemWidth::Double,
                PageAddressSpace::MainMemory(MEMORY_AS),
            );
            return;
        }

        ctx.emit_checked_call_without_page_flush("openvm_hint_prepare", &[&ptr, "1u"]);
        ctx.reserve_preflight_timestamp_slots("2u");
        ctx.reserve_replay_values("1u");
        ctx.write_line("uint64_t hint_word;");
        ctx.emit_call_without_page_flush("openvm_hint_read_words", &["&hint_word", "1u"]);
        ctx.advance_timestamp(1);
        ctx.write_aligned_mem_block(&ptr, "hint_word");
        ctx.append_replay_value("hint_word");
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

/// HINT_BUFFER: pop `8 * reg[num_words_reg]` bytes from the hint stream and
/// write them sequentially starting at `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub(crate) struct HintBufferInstr {
    pub(crate) ptr_reg: Reg,
    pub(crate) num_words_reg: Reg,
    pub(crate) chip_idx: Option<AirIndex>,
}

impl ExtInstr for HintBufferInstr {
    fn opname(&self) -> &str {
        "hint_buffer"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_var(self.ptr_reg);
        let n = ctx.read_var(self.num_words_reg);
        ctx.write_line(&format!(
            "if (unlikely(({n} - 1ull) >= {MAX_HINT_BUFFER_DWORDS}ull)) {{"
        ));
        ctx.emit_trap();
        ctx.write_line("}");
        let callback_count = format!("(uint32_t)({n})");
        if ctx.is_preflight() {
            ctx.emit_checked_call_without_page_flush(
                "openvm_hint_prepare",
                &[&ptr, &callback_count],
            );
            ctx.reserve_preflight_timestamp_slots(&format!("((uint32_t)({n}) * 3u - 2u)"));
            ctx.reserve_replay_values(&callback_count);
            ctx.write_line(&format!("uint64_t hint_words[{MAX_HINT_BUFFER_DWORDS}u];"));
            ctx.emit_call_without_page_flush(
                "openvm_hint_read_words",
                &["hint_words", &callback_count],
            );
            ctx.write_line(&format!(
                "for (uint32_t hint_idx = 0u; hint_idx < (uint32_t)({n}); ++hint_idx) {{"
            ));
            ctx.write_line("if (hint_idx != 0u) {");
            ctx.advance_timestamp(2);
            ctx.write_line("}");
            ctx.write_aligned_mem_block(
                &format!("({ptr} + (uint64_t)hint_idx * 8ull)"),
                "hint_words[hint_idx]",
            );
            ctx.append_replay_value("hint_words[hint_idx]");
            ctx.write_line("}");
        } else {
            ctx.reserve_replay_values(&callback_count);
            ctx.emit_checked_call_without_page_flush(
                "openvm_hint_buffer",
                &[&ptr, &callback_count],
            );
            ctx.append_replay_memory_u64_range(&ptr, &callback_count);
            ctx.trace_page_access_u64_range(&ptr, &n, PageAddressSpace::MainMemory(MEMORY_AS));
        }
        // Block entry credits one row; runtime metering adds the remaining
        // `(n - 1)` rows.
        let chip_idx = air_index_to_c(self.chip_idx);
        // After the check above, n - 1 is at most 1022.
        ctx.trace_chip_if_nonzero(chip_idx, &format!("(uint32_t)({n} - 1ull)"));
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

/// Store `src_reg` at `ptr_reg + offset` in the public-values address space.
#[derive(Debug, Clone)]
pub(crate) struct RevealInstr {
    pub(crate) src_reg: Reg,
    pub(crate) ptr_reg: Reg,
    pub(crate) offset: i32,
}

impl ExtInstr for RevealInstr {
    fn opname(&self) -> &str {
        "reveal"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_var(self.ptr_reg);
        let src = ctx.read_var(self.src_reg);
        let addr = match self.offset.cmp(&0) {
            std::cmp::Ordering::Less => {
                format!("({ptr} - 0x{:08x}ull)", self.offset.unsigned_abs())
            }
            std::cmp::Ordering::Equal => ptr.clone(),
            std::cmp::Ordering::Greater => format!("({ptr} + 0x{:08x}ull)", self.offset),
        };
        // The callback emits the proof-visible public-values memory events.
        // Reserve only their logical clock here so compact checkpoint
        // preflight can preserve the schedule without logging those events.
        // Full preflight performs the aligned block-write reservation inside the callback.
        ctx.reserve_preflight_timestamp_slots("1u");
        ctx.emit_checked_call("openvm_reveal", &["state", &src, &ptr, &addr]);
        ctx.trace_page_access(
            &addr,
            MemWidth::Double,
            PageAddressSpace::Other(PUBLIC_VALUES_AS),
        );
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

/// RVR extension for RV64 IO hint-store and REVEAL instructions.
pub struct Rv64IoExtension {
    hint_store_chip_idx: Option<AirIndex>,
}

impl Rv64IoExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let hint_store_chip_idx = opcode_air_idx(ctx, HintStoreOpcode::HINT_STORED)?;
        Ok(Self {
            hint_store_chip_idx,
        })
    }
}

impl RvrExtension for Rv64IoExtension {
    fn try_lift(&self, insn: &Instruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == HintStoreOpcode::HINT_STORED.global_opcode_usize() {
            if insn.d.as_u32() != REGISTER_AS || insn.e.as_u32() != MEMORY_AS {
                return None;
            }
            let ptr_reg = decode_reg(insn.b.as_u32());
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(HintStoreWInstr { ptr_reg }),
                source_loc: None,
            }));
        }

        if opcode == HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
            if insn.d.as_u32() != REGISTER_AS || insn.e.as_u32() != MEMORY_AS {
                return None;
            }
            let num_words_reg = decode_reg(insn.a.as_u32());
            let ptr_reg = decode_reg(insn.b.as_u32());
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(HintBufferInstr {
                    ptr_reg,
                    num_words_reg,
                    chip_idx: self.hint_store_chip_idx,
                }),
                source_loc: None,
            }));
        }

        if let Some(reveal) = decode_reveal(insn) {
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(reveal),
                source_loc: None,
            }));
        }

        None
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64io_callbacks.h",
            include_str!("../c/rv64io_callbacks.h"),
        )]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64io_callbacks.c",
            include_str!("../c/rv64io_callbacks.c"),
        )]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        IO_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
    }
}

pub struct Rv64IoRuntimeHooks;

impl RvrRuntimeExtension for Rv64IoRuntimeHooks {
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        let register_fn: RegisterRv64IoHostCallbacksFn = unsafe {
            let sym = lib
                .get::<RegisterRv64IoHostCallbacksFn>(b"register_rv64io_host_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        let callbacks = Rv64IoHostCallbacks {
            hint_prepare: host_hint_prepare,
            hint_read_words: host_hint_read_words,
            hint_storew: host_hint_storew,
            hint_buffer: host_hint_buffer,
            reveal_prepare: host_reveal_prepare,
            reveal_commit: host_reveal_commit,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

fn decode_reveal(insn: &Instruction) -> Option<RevealInstr> {
    let opcode = insn.opcode.as_usize();
    let src_reg_ptr = insn.a.checked_as_u32()?;
    let base_reg_ptr = insn.b.checked_as_u32()?;
    let immediate = insn.c.checked_as_u32()?;
    let src_address_space = insn.d.checked_as_u32()?;
    let dst_address_space = insn.e.checked_as_u32()?;
    let is_enabled = insn.f.checked_as_u32()?;
    let immediate_sign = insn.g.checked_as_u32()?;

    if opcode != RevealOpcode::REVEAL.global_opcode_usize()
        || !is_valid_register_pointer(src_reg_ptr)
        || !is_valid_register_pointer(base_reg_ptr)
        || immediate > u16::MAX as u32
        || src_address_space != REGISTER_AS
        || dst_address_space != PUBLIC_VALUES_AS
        || is_enabled != 1
        || immediate_sign > 1
    {
        return None;
    }

    Some(RevealInstr {
        src_reg: decode_reg(src_reg_ptr),
        ptr_reg: decode_reg(base_reg_ptr),
        offset: decode_imm_cg(insn) as i32,
    })
}

type RegisterRv64IoHostCallbacksFn = unsafe extern "C" fn(*const Rv64IoHostCallbacks);

/// Host callback table shared with `rv64io_callbacks.c`.
#[repr(C)]
struct Rv64IoHostCallbacks {
    hint_prepare: extern "C" fn(*mut c_void, u64, u32) -> bool,
    hint_read_words: unsafe extern "C" fn(*mut c_void, *mut u64, u32),
    hint_storew: extern "C" fn(*mut c_void, u64) -> bool,
    hint_buffer: extern "C" fn(*mut c_void, u64, u32) -> bool,
    reveal_prepare: extern "C" fn(*mut c_void, u64, u64, u64, *mut RevealPlan) -> bool,
    reveal_commit: unsafe extern "C" fn(*mut c_void, *const RevealPlan),
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RevealPlan {
    address: u64,
    previous: u64,
    post: u64,
}

fn checked_hint_range(
    io: &OpenVmIoState<'_>,
    dest_addr: u64,
    num_words: u32,
) -> Option<Range<usize>> {
    let num_words = num_words as usize;
    if num_words == 0
        || num_words > MAX_HINT_BUFFER_DWORDS
        || !dest_addr.is_multiple_of(REGISTER_BYTES)
        || io.memory_ptr.is_null()
    {
        return None;
    }
    let num_bytes = num_words.checked_mul(REGISTER_NUM_LIMBS)?;
    if io.hint_stream.remaining() < num_bytes {
        return None;
    }
    checked_mem_bounds_range(dest_addr, num_bytes as u64)
}

/// Validate a hint-store operation without consuming hints or mutating memory.
extern "C" fn host_hint_prepare(ctx: *mut c_void, dest_addr: u64, num_words: u32) -> bool {
    let io = unsafe { &*(ctx as *mut OpenVmIoState<'_>) };
    checked_hint_range(io, dest_addr, num_words).is_some()
}

/// Consume validated hint words into a host buffer.
///
/// # Safety
///
/// `words` must point to writable storage for `num_words` elements, and
/// [`host_hint_prepare`] must have succeeded for the same word count without
/// an intervening hint-stream mutation.
unsafe extern "C" fn host_hint_read_words(ctx: *mut c_void, words: *mut u64, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let words = unsafe { std::slice::from_raw_parts_mut(words, num_words as usize) };
    for word in words {
        let mut bytes = [0; REGISTER_NUM_LIMBS];
        io.hint_stream.copy_to_slice(&mut bytes);
        *word = u64::from_le_bytes(bytes);
    }
}

/// Validate and copy one hint word directly into guest memory.
extern "C" fn host_hint_storew(ctx: *mut c_void, dest_addr: u64) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if io.hint_stream.remaining() < REGISTER_NUM_LIMBS
        || !dest_addr.is_multiple_of(REGISTER_BYTES)
        || io.memory_ptr.is_null()
    {
        return false;
    }
    let Some(range) = checked_mem_bounds_range(dest_addr, REGISTER_BYTES) else {
        return false;
    };
    let dst =
        unsafe { std::slice::from_raw_parts_mut(io.memory_ptr.add(range.start), range.len()) };
    io.hint_stream.copy_to_slice(dst);
    true
}

/// Validate and copy hint words directly into guest memory.
extern "C" fn host_hint_buffer(ctx: *mut c_void, dest_addr: u64, num_words: u32) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let Some(range) = checked_hint_range(io, dest_addr, num_words) else {
        return false;
    };
    let dst =
        unsafe { std::slice::from_raw_parts_mut(io.memory_ptr.add(range.start), range.len()) };
    io.hint_stream.copy_to_slice(dst);
    true
}

/// Validate and materialize one complete pre/post AS3 block without mutation.
extern "C" fn host_reveal_prepare(
    ctx: *mut c_void,
    src_val: u64,
    base_addr: u64,
    effective_addr: u64,
    plan: *mut RevealPlan,
) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if base_addr > u32::MAX as u64 || effective_addr > u32::MAX as u64 || plan.is_null() {
        return false;
    }
    let width = WORD_SIZE;
    let Some(end) = effective_addr.checked_add(width as u64) else {
        return false;
    };
    if end > io.public_values.len() as u64 {
        return false;
    }
    if !effective_addr.is_multiple_of(WORD_SIZE as u64)
        || u32::try_from(effective_addr >> 1).is_err()
    {
        return false;
    }
    let start = effective_addr as usize;
    let previous = u64::from_le_bytes(io.public_values[start..start + width].try_into().unwrap());
    unsafe {
        plan.write(RevealPlan {
            address: effective_addr,
            previous,
            post: src_val,
        });
    }
    true
}

/// Apply a previously validated AS3 plan. Generated C logs every block first.
///
/// # Safety
///
/// `plan` must point to a plan produced by [`host_reveal_prepare`] for this IO
/// context, with no intervening AS3 mutation.
unsafe extern "C" fn host_reveal_commit(ctx: *mut c_void, plan: *const RevealPlan) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let plan = unsafe { &*plan };
    let start = plan.address as usize;
    io.public_values[start..start + WORD_SIZE].copy_from_slice(&plan.post.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, ptr::null_mut};

    use openvm_circuit::arch::HintStream;
    use openvm_instructions::SystemOpcode;
    use openvm_riscv_transpiler::LoadStoreOpcode;
    use rand::{rngs::StdRng, SeedableRng};
    use test_case::test_case;

    use super::*;
    use crate::phantom::host_hint_input;

    #[derive(Default)]
    struct TestEmitCtx {
        lines: Vec<String>,
        preflight: bool,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn is_preflight(&self) -> bool {
            self.preflight
        }

        fn read_var(&mut self, var: Reg) -> String {
            format!("r{}", var.index())
        }

        fn peek_var(&mut self, var: Reg) -> String {
            format!("r{}", var.index())
        }

        fn advance_timestamp(&mut self, slots: u32) {
            self.write_line(&format!("advance_timestamp({slots});"));
        }

        fn write_var(&mut self, _var: Reg, _val: &str) {}

        fn write_line(&mut self, s: &str) {
            self.lines.push(s.to_string());
        }

        fn emit_trap(&mut self) {
            self.write_line("trap;");
        }

        fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!(
                "uint32_t {tmp} = read_mem({base}, {offset}, {width}, {signed});"
            ));
            tmp
        }

        fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
            self.write_line(&format!("write_mem({base}, {offset}, {val}, {width});"));
        }

        fn write_aligned_mem_block(&mut self, addr: &str, val: &str) {
            self.write_line(&format!("write_aligned_mem_block({addr}, {val});"));
        }

        fn reserve_preflight_timestamp_slots(&mut self, slots: &str) {
            self.write_line(&format!("reserve_preflight_timestamp_slots({slots});"));
        }

        fn reserve_replay_values(&mut self, count: &str) {
            self.write_line(&format!("reserve_replay_values({count});"));
        }

        fn append_replay_value(&mut self, value: &str) {
            self.write_line(&format!("append_replay_value({value});"));
        }

        fn emit_call(&mut self, name: &str, args: &[&str]) {
            self.write_line(&format!("{name}({});", args.join(", ")));
        }

        fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
            self.write_line(&format!("{name}({});", args.join(", ")));
        }

        fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!("{ret_ty} {tmp} = {name}({});", args.join(", ")));
            tmp
        }

        fn emit_call_with_trace_result(
            &mut self,
            ret_ty: &str,
            name: &str,
            args: &[&str],
        ) -> Option<String> {
            Some(self.emit_call_expr(ret_ty, name, args))
        }

        fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
            self.write_line(&format!("trace_chip(state, {chip_idx}u, {count_expr});"));
        }

        fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str) {
            self.write_line(&format!("if (({count_expr}) != 0u) {{"));
            self.trace_chip(chip_idx, count_expr);
            self.write_line("}");
        }

        fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace) {
            let size = width.bytes();
            self.write_line(&format!(
                "trace_page_access(state, {addr}, {size}u, {}u);",
                addr_space.id()
            ));
        }

        fn trace_page_access_u64_range(
            &mut self,
            base_addr: &str,
            num_dwords: &str,
            addr_space: PageAddressSpace,
        ) {
            self.write_line(&format!(
                "trace_page_access_u64_range(state, {base_addr}, {num_dwords}, {}u);",
                addr_space.id()
            ));
        }
    }

    #[test]
    fn rv64io_lifts_reveal_as_a_doubleword_public_values_write() {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = Instruction::from_usize(
            RevealOpcode::REVEAL.global_opcode(),
            [
                8,
                16,
                0,
                REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        );
        let lifted = ext.try_lift(&inst, 0x100).unwrap();
        let LiftedInstr::Body(InstrAt { instr, .. }) = lifted else {
            panic!("expected reveal body instruction");
        };

        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);
        assert_eq!(ctx.lines[0], "reserve_preflight_timestamp_slots(1u);");
        assert_eq!(
            ctx.lines[1],
            "bool tmp1 = openvm_reveal(state, r1, r2, r2);"
        );
        assert_eq!(ctx.lines[2], "if (unlikely(!tmp1)) {");
        assert_eq!(
            ctx.lines[5],
            format!("trace_page_access(state, r2, 8u, {PUBLIC_VALUES_AS}u);")
        );
    }

    #[test_case(LoadStoreOpcode::STORED; "dword")]
    #[test_case(LoadStoreOpcode::STOREW; "word")]
    #[test_case(LoadStoreOpcode::STOREH; "halfword")]
    #[test_case(LoadStoreOpcode::STOREB; "byte")]
    fn rv64io_does_not_treat_load_store_opcodes_as_reveal(opcode: LoadStoreOpcode) {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = Instruction::from_usize(
            opcode.global_opcode(),
            [
                8,
                16,
                0,
                REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        );

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test_case(PUBLIC_VALUES_AS, PUBLIC_VALUES_AS; "source_domain")]
    #[test_case(REGISTER_AS, MEMORY_AS; "destination_domain")]
    fn rv64io_rejects_reveal_with_invalid_address_spaces(d: u32, e: u32) {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = Instruction::from_usize(
            RevealOpcode::REVEAL.global_opcode(),
            [8, 16, 0, d as usize, e as usize, 1, 0],
        );

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test_case([7, 16, 0, REGISTER_AS, PUBLIC_VALUES_AS, 1, 0]; "unaligned_source")]
    #[test_case([u8::MAX as u32 + 1, 16, 0, REGISTER_AS, PUBLIC_VALUES_AS, 1, 0]; "source_out_of_range")]
    #[test_case([8, 15, 0, REGISTER_AS, PUBLIC_VALUES_AS, 1, 0]; "unaligned_base")]
    #[test_case([8, u8::MAX as u32 + 1, 0, REGISTER_AS, PUBLIC_VALUES_AS, 1, 0]; "base_out_of_range")]
    #[test_case([8, 16, u16::MAX as u32 + 1, REGISTER_AS, PUBLIC_VALUES_AS, 1, 0]; "immediate_out_of_range")]
    #[test_case([8, 16, 0, REGISTER_AS, PUBLIC_VALUES_AS, 0, 0]; "invalid_f")]
    #[test_case([8, 16, 0, REGISTER_AS, PUBLIC_VALUES_AS, 1, 2]; "invalid_sign")]
    fn rv64io_rejects_malformed_reveal_operands(operands: [u32; 7]) {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = Instruction::from_usize(
            RevealOpcode::REVEAL.global_opcode(),
            operands.map(|operand| operand as usize),
        );

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test]
    fn hint_stores_require_register_and_memory_address_spaces() {
        let ext = Rv64IoExtension::new(None).unwrap();
        for opcode in [HintStoreOpcode::HINT_STORED, HintStoreOpcode::HINT_BUFFER] {
            let valid = Instruction::from_usize(
                opcode.global_opcode(),
                [8, 16, 0, REGISTER_AS as usize, MEMORY_AS as usize, 0, 0],
            );
            assert!(ext.try_lift(&valid, 0x100).is_some());

            let invalid_d = Instruction::from_usize(
                opcode.global_opcode(),
                [8, 16, 0, MEMORY_AS as usize, MEMORY_AS as usize, 0, 0],
            );
            assert!(ext.try_lift(&invalid_d, 0x100).is_none());

            let invalid_e = Instruction::from_usize(
                opcode.global_opcode(),
                [
                    8,
                    16,
                    0,
                    REGISTER_AS as usize,
                    PUBLIC_VALUES_AS as usize,
                    0,
                    0,
                ],
            );
            assert!(ext.try_lift(&invalid_e, 0x100).is_none());
        }
    }

    #[test]
    fn rv64io_ignores_non_store_public_values_shaped_instruction() {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = Instruction::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [
                8,
                16,
                0,
                REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        );

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test]
    fn reveal_accounts_for_the_offset_public_values_page() {
        let mut ctx = TestEmitCtx::default();
        let instr = RevealInstr {
            src_reg: Reg::new(5),
            ptr_reg: Reg::new(10),
            offset: 12,
        };
        instr.emit_c(&mut ctx);

        assert_eq!(ctx.lines[0], "reserve_preflight_timestamp_slots(1u);");
        assert_eq!(
            ctx.lines[1],
            "bool tmp1 = openvm_reveal(state, r5, r10, (r10 + 0x0000000cull));"
        );
        assert_eq!(ctx.lines[2], "if (unlikely(!tmp1)) {");
        assert_eq!(
            ctx.lines[5],
            format!("trace_page_access(state, (r10 + 0x0000000cull), 8u, {PUBLIC_VALUES_AS}u);")
        );
    }

    #[test]
    fn hint_storew_emits_the_three_slot_schedule() {
        let instr = HintStoreWInstr {
            ptr_reg: Reg::new(5),
        };

        let mut ctx = TestEmitCtx {
            preflight: true,
            ..Default::default()
        };
        instr.emit_c(&mut ctx);

        assert_eq!(
            ctx.lines,
            [
                "if (unlikely(!openvm_hint_prepare(r5, 1u))) {",
                "trap;",
                "}",
                "reserve_preflight_timestamp_slots(2u);",
                "reserve_replay_values(1u);",
                "uint64_t hint_word;",
                "openvm_hint_read_words(&hint_word, 1u);",
                "advance_timestamp(1);",
                "write_aligned_mem_block(r5, hint_word);",
                "append_replay_value(hint_word);",
            ]
        );
    }

    #[test]
    fn hint_storew_uses_bulk_transfer_outside_preflight() {
        let instr = HintStoreWInstr {
            ptr_reg: Reg::new(5),
        };
        let mut ctx = TestEmitCtx::default();

        instr.emit_c(&mut ctx);

        let emitted = ctx.lines.join("\n");
        assert!(emitted.contains("openvm_hint_storew(r5)"));
        assert!(emitted.contains("trace_page_access"));
        assert!(!emitted.contains("hint_word"));
        assert!(!emitted.contains("openvm_hint_read_words"));
    }

    #[test]
    fn hint_buffer_emits_validation_reservation_and_three_slots_per_word() {
        let instr = HintBufferInstr {
            ptr_reg: Reg::new(5),
            num_words_reg: Reg::new(6),
            chip_idx: None,
        };

        let mut ctx = TestEmitCtx {
            preflight: true,
            ..Default::default()
        };
        instr.emit_c(&mut ctx);

        let emitted = ctx.lines.join("\n");
        assert!(emitted.contains("if (unlikely((r6 - 1ull) >= 1023ull)) {"));
        assert!(emitted.contains("if (unlikely(!openvm_hint_prepare(r5, (uint32_t)(r6)))) {"));
        assert!(emitted.contains("reserve_preflight_timestamp_slots(((uint32_t)(r6) * 3u - 2u));"));
        assert!(emitted.contains("reserve_replay_values((uint32_t)(r6));"));
        assert!(emitted.contains("uint64_t hint_words[1023u];"));
        assert!(emitted.contains("openvm_hint_read_words(hint_words, (uint32_t)(r6));"));
        assert!(emitted
            .contains("for (uint32_t hint_idx = 0u; hint_idx < (uint32_t)(r6); ++hint_idx) {"));
        assert!(emitted.contains("if (hint_idx != 0u) {"));
        assert!(emitted.contains("advance_timestamp(2);"));
        assert!(emitted.contains(
            "write_aligned_mem_block((r5 + (uint64_t)hint_idx * 8ull), hint_words[hint_idx]);"
        ));
        assert!(emitted.contains("append_replay_value(hint_words[hint_idx]);"));
        assert!(!emitted.contains("trace_page_access"));
    }

    #[test]
    fn hint_buffer_uses_bulk_transfer_outside_preflight() {
        let instr = HintBufferInstr {
            ptr_reg: Reg::new(5),
            num_words_reg: Reg::new(6),
            chip_idx: None,
        };
        let mut ctx = TestEmitCtx::default();

        instr.emit_c(&mut ctx);

        let emitted = ctx.lines.join("\n");
        assert!(emitted.contains("openvm_hint_buffer(r5, (uint32_t)(r6))"));
        assert!(emitted.contains("trace_page_access_u64_range"));
        assert!(!emitted.contains("uint64_t hint_words"));
        assert!(!emitted.contains("openvm_hint_read_words"));
        assert!(!emitted.contains("for (uint32_t hint_idx"));
    }

    #[test_case(0x1122_3344_5566_7788, 0; "first_dword")]
    #[test_case(0xaabb_ccdd_eeff_0123, 8; "last_dword")]
    fn host_reveal_plans_then_commits_doubleword(src_val: u64, addr: u64) {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 16];
        let mut public_values = vec![0u8; 16];
        let mut deferrals = Vec::new();

        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };

        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;
        let mut plan = RevealPlan::default();
        assert!(host_reveal_prepare(ctx, src_val, addr, addr, &mut plan));
        assert!(io.public_values.iter().all(|&byte| byte == 0));
        unsafe { host_reveal_commit(ctx, &plan) };

        let start = addr as usize;
        let end = start + WORD_SIZE;
        assert_eq!(&io.public_values[start..end], &src_val.to_le_bytes());
        assert!(io.public_values[..start].iter().all(|&byte| byte == 0));
        assert!(io.public_values[end..].iter().all(|&byte| byte == 0));
    }

    #[test_case(u64::MAX; "address_overflow")]
    #[test_case(1; "misaligned")]
    #[test_case(9; "out_of_bounds")]
    fn host_reveal_rejects_invalid_range(addr: u64) {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 16];
        let mut public_values = vec![0u8; 16];
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };

        let mut plan = RevealPlan::default();
        assert!(!host_reveal_prepare(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            u64::MAX,
            addr,
            addr,
            &mut plan,
        ));
        assert!(io.public_values.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn host_reveal_rejects_non_u32_base_or_effective_address_without_mutation() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 16];
        let mut public_values = vec![0u8; 16];
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let mut plan = RevealPlan::default();

        assert!(!host_reveal_prepare(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            1,
            u64::from(u32::MAX) + 1,
            0,
            &mut plan,
        ));
        // A positive offset from the largest valid base must not produce an
        // effective byte address outside the AIR's u32 pointer domain.
        assert!(!host_reveal_prepare(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            1,
            u64::from(u32::MAX),
            u64::from(u32::MAX) + 1,
            &mut plan,
        ));
        assert!(io.public_values.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn host_hint_callbacks_support_materialized_and_bulk_transfers() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        hint_stream.set_hint((10u8..22).collect());

        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0xa5; 16];
        let original_memory = memory.clone();
        let mut public_values = vec![];
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;

        assert!(host_hint_prepare(ctx, 0, 1));
        assert_eq!(io.hint_stream.remaining(), 12);
        assert_eq!(memory, original_memory);

        let mut words = [0u64; 1];
        unsafe { host_hint_read_words(ctx, words.as_mut_ptr(), 1) };

        assert_eq!(
            words,
            [u64::from_le_bytes([10, 11, 12, 13, 14, 15, 16, 17])]
        );
        assert_eq!(io.hint_stream.remaining(), 4);
        assert_eq!(memory, original_memory);

        let word_hint = (40u8..48).collect::<Vec<_>>();
        io.hint_stream.set_hint(word_hint.clone());
        assert!(host_hint_storew(ctx, 8));
        assert_eq!(&memory[8..], word_hint);
        assert_eq!(io.hint_stream.remaining(), 0);

        let bulk_hint = (20u8..36).collect::<Vec<_>>();
        io.hint_stream.set_hint(bulk_hint.clone());
        assert!(host_hint_buffer(ctx, 0, 2));
        assert_eq!(memory, bulk_hint);
        assert_eq!(io.hint_stream.remaining(), 0);
    }

    #[test]
    fn host_input_callbacks_expose_length_payload_and_padding_as_words() {
        let payload = (1u8..=9).collect::<Vec<_>>();
        let mut input_stream = VecDeque::from([payload.clone()]);
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0xa5; 24];
        let mut public_values = vec![];
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;

        assert!(host_hint_input(ctx));
        assert!(host_hint_prepare(ctx, 0, 3));
        let mut words = [0u64; 3];
        unsafe { host_hint_read_words(ctx, words.as_mut_ptr(), 3) };

        assert_eq!(words[0], payload.len() as u64);
        assert_eq!(words[1].to_le_bytes(), [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(words[2].to_le_bytes(), [9, 0, 0, 0, 0, 0, 0, 0]);
        assert!(memory.iter().all(|&byte| byte == 0xa5));
        assert_eq!(io.hint_stream.remaining(), 0);
        assert!(io.input_stream.is_empty());
    }

    #[test]
    fn host_hint_prepare_rejects_invalid_operations_without_mutation() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        hint_stream.set_hint(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0xa5; 16];
        let original_memory = memory.clone();
        let mut public_values = Vec::new();
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: std::ptr::null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;

        assert!(!host_hint_prepare(ctx, 1, 0));
        // One full advice word is available, so this fails only on alignment.
        assert!(!host_hint_prepare(ctx, 1, 1));
        assert!(!host_hint_prepare(
            ctx,
            0,
            (MAX_HINT_BUFFER_DWORDS + 1) as u32
        ));
        assert!(!host_hint_prepare(ctx, u64::MAX - 7, 1));
        assert_eq!(io.hint_stream.remaining(), 8);
        let mut hint = [0; 8];
        io.hint_stream.copy_to_slice(&mut hint);
        assert_eq!(hint, [1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(memory, original_memory);
    }

    #[test]
    fn host_hint_prepare_accepts_the_maximum_word_count() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let hint = (0..MAX_HINT_BUFFER_DWORDS * REGISTER_NUM_LIMBS)
            .map(|i| i as u8)
            .collect::<Vec<_>>();
        hint_stream.set_hint(hint.clone());
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 8];
        let mut public_values = Vec::new();
        let mut deferrals = Vec::new();
        let mut io = OpenVmIoState {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: null_mut(),
            deferral_memory_len_bytes: 0,
            preflight_deferral_dirty_pages: None,
            deferrals: &mut deferrals,
        };
        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;

        assert!(host_hint_prepare(ctx, 0, MAX_HINT_BUFFER_DWORDS as u32));
        let mut words = vec![0u64; MAX_HINT_BUFFER_DWORDS];
        unsafe {
            host_hint_read_words(ctx, words.as_mut_ptr(), MAX_HINT_BUFFER_DWORDS as u32);
        }

        assert_eq!(io.hint_stream.remaining(), 0);
        assert_eq!(words[0].to_le_bytes(), hint[..8]);
        assert_eq!(
            words[MAX_HINT_BUFFER_DWORDS - 1].to_le_bytes(),
            hint[hint.len() - 8..]
        );
    }
}
