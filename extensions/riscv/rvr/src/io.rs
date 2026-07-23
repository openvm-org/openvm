//! RV64 IO instruction lifting and host callbacks.

use std::ffi::c_void;

use openvm_circuit::arch::rvr::io::{checked_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{
    riscv::{RV64_REGISTER_AS, RV64_REGISTER_BYTES, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, PUBLIC_VALUES_AS,
};
use openvm_platform::WORD_SIZE;
use openvm_riscv_transpiler::{Rv64HintStoreOpcode, Rv64LoadStoreOpcode, MAX_HINT_BUFFER_DWORDS};
use rvr_openvm_ir::{
    CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr, MemWidth, PageAddressSpace,
};
use rvr_openvm_lift::{
    air_index_to_c, max_main_memory_pages_for_contiguous_range, opcode_air_idx, AirIndex,
    ExtensionError, RvrExtension, RvrExtensionCtx, RvrInstruction, RvrRuntimeExtension,
};

use crate::instruction::{decode_imm_cg, decode_reg, Reg};

// HINT_BUFFER writes the maximum hint payload as one contiguous range.
const RV64_IO_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION: usize =
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
        ctx.emit_checked_call_without_page_flush("openvm_hint_prepare", &[&ptr, "1u"]);
        ctx.reserve_preflight_writes("1u", "2u");
        ctx.write_line("uint64_t hint_word;");
        ctx.emit_call_without_page_flush("openvm_hint_read_words", &["&hint_word", "1u"]);
        ctx.advance_timestamp(1);
        ctx.write_aligned_mem_block(&ptr, "hint_word");
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
        ctx.emit_checked_call_without_page_flush("openvm_hint_prepare", &[&ptr, &callback_count]);
        ctx.reserve_preflight_writes(&callback_count, &format!("((uint32_t)({n}) * 3u - 2u)"));
        ctx.write_line(&format!("uint64_t hint_words[{MAX_HINT_BUFFER_DWORDS}u];"));
        ctx.emit_call_without_page_flush(
            "openvm_hint_read_words",
            &["hint_words", &callback_count],
        );
        ctx.write_aligned_mem_block(&ptr, "hint_words[0]");
        ctx.write_line(&format!(
            "for (uint32_t hint_idx = 1u; hint_idx < (uint32_t)({n}); ++hint_idx) {{"
        ));
        ctx.advance_timestamp(2);
        ctx.write_aligned_mem_block(
            &format!("({ptr} + (uint64_t)hint_idx * 8ull)"),
            "hint_words[hint_idx]",
        );
        ctx.write_line("}");
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

/// Store the low `width` bytes of `src_reg` at `ptr_reg + offset` in the
/// public-values address space. This node also implements REVEAL.
#[derive(Debug, Clone)]
pub(crate) struct RevealInstr {
    pub(crate) src_reg: Reg,
    pub(crate) ptr_reg: Reg,
    pub(crate) offset: i32,
    pub(crate) width: MemWidth,
}

impl ExtInstr for RevealInstr {
    fn opname(&self) -> &str {
        "reveal"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let src = ctx.read_var(self.src_reg);
        let ptr = ctx.read_var(self.ptr_reg);
        let addr = match self.offset.cmp(&0) {
            std::cmp::Ordering::Less => {
                format!("({ptr} - 0x{:08x}ull)", self.offset.unsigned_abs())
            }
            std::cmp::Ordering::Equal => ptr,
            std::cmp::Ordering::Greater => format!("({ptr} + 0x{:08x}ull)", self.offset),
        };
        let width = format!("{}u", self.width.bytes());
        ctx.emit_checked_call_without_page_flush("openvm_reveal", &[&src, &addr, &width]);
        ctx.trace_page_access(&addr, self.width, PageAddressSpace::Other(PUBLIC_VALUES_AS));
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// RVR extension for RV64 IO hint-store instructions and stores to public
/// values, including REVEAL.
pub struct Rv64IoExtension {
    hint_store_chip_idx: Option<AirIndex>,
}

impl Rv64IoExtension {
    pub fn new(ctx: Option<&RvrExtensionCtx>) -> Result<Self, ExtensionError> {
        let hint_store_chip_idx = opcode_air_idx(ctx, Rv64HintStoreOpcode::HINT_STORED)?;
        Ok(Self {
            hint_store_chip_idx,
        })
    }
}

impl RvrExtension for Rv64IoExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Rv64HintStoreOpcode::HINT_STORED.global_opcode_usize() {
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(HintStoreWInstr { ptr_reg }),
                source_loc: None,
            }));
        }

        if opcode == Rv64HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
            let num_words_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
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

        if let Some(width) = public_values_store_width(insn) {
            let src_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            let offset = decode_imm_cg(insn);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(RevealInstr {
                    src_reg,
                    ptr_reg,
                    offset: offset as i32,
                    width,
                }),
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
        RV64_IO_MAX_MAIN_MEMORY_PAGES_PER_INSTRUCTION
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
            reveal: host_reveal,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

fn public_values_store_width(insn: &RvrInstruction) -> Option<MemWidth> {
    if insn.d != RV64_REGISTER_AS || insn.e != PUBLIC_VALUES_AS {
        return None;
    }

    match insn.opcode.as_usize() {
        opcode if opcode == Rv64LoadStoreOpcode::STORED.global_opcode_usize() => {
            Some(MemWidth::Double)
        }
        opcode if opcode == Rv64LoadStoreOpcode::STOREW.global_opcode_usize() => {
            Some(MemWidth::Word)
        }
        opcode if opcode == Rv64LoadStoreOpcode::STOREH.global_opcode_usize() => {
            Some(MemWidth::Half)
        }
        opcode if opcode == Rv64LoadStoreOpcode::STOREB.global_opcode_usize() => {
            Some(MemWidth::Byte)
        }
        _ => None,
    }
}

type RegisterRv64IoHostCallbacksFn = unsafe extern "C" fn(*const Rv64IoHostCallbacks);

/// Host callback table shared with `rv64io_callbacks.c`.
#[repr(C)]
struct Rv64IoHostCallbacks {
    hint_prepare: extern "C" fn(*mut c_void, u64, u32) -> bool,
    hint_read_words: unsafe extern "C" fn(*mut c_void, *mut u64, u32),
    reveal: extern "C" fn(*mut c_void, u64, u64, u8) -> bool,
}

/// Validate a hint-store operation without consuming hints or mutating memory.
extern "C" fn host_hint_prepare(ctx: *mut c_void, dest_addr: u64, num_words: u32) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let num_words = num_words as usize;
    if num_words == 0
        || num_words > MAX_HINT_BUFFER_DWORDS
        || !dest_addr.is_multiple_of(RV64_REGISTER_BYTES)
        || io.memory_ptr.is_null()
    {
        return false;
    }
    let Some(num_bytes) = num_words.checked_mul(RV64_REGISTER_NUM_LIMBS) else {
        return false;
    };
    io.hint_stream.remaining() >= num_bytes
        && checked_mem_bounds_range(dest_addr, num_bytes as u64).is_some()
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
        let mut bytes = [0; RV64_REGISTER_NUM_LIMBS];
        io.hint_stream.copy_to_slice(&mut bytes);
        *word = u64::from_le_bytes(bytes);
    }
}

/// Host callback for stores to `PUBLIC_VALUES_AS`.
/// Generated C adds the trace-row cost.
extern "C" fn host_reveal(ctx: *mut c_void, src_val: u64, addr: u64, width: u8) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let width = match width {
        1 | 2 | 4 | 8 => usize::from(width),
        _ => return false,
    };
    let Some(end) = addr.checked_add(width as u64) else {
        return false;
    };
    if end > io.public_values.len() as u64 {
        return false;
    }
    let range = addr as usize..end as usize;
    io.public_values[range].copy_from_slice(&src_val.to_le_bytes()[..width]);
    true
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, ptr::null_mut};

    use openvm_circuit::arch::HintStream;
    use openvm_instructions::{instruction::Instruction, SystemOpcode};
    use p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};
    use test_case::test_case;

    use super::*;
    use crate::phantom::host_hint_input;

    #[derive(Default)]
    struct TestEmitCtx {
        lines: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
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

        fn reserve_preflight_writes(&mut self, writes: &str, slots: &str) {
            self.write_line(&format!("reserve_preflight_writes({writes}, {slots});"));
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

    #[test_case(Rv64LoadStoreOpcode::STORED, 8; "dword")]
    #[test_case(Rv64LoadStoreOpcode::STOREW, 4; "word")]
    #[test_case(Rv64LoadStoreOpcode::STOREH, 2; "halfword")]
    #[test_case(Rv64LoadStoreOpcode::STOREB, 1; "byte")]
    fn rv64io_lifts_public_values_store_width(opcode: Rv64LoadStoreOpcode, width: u8) {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            opcode.global_opcode(),
            [
                8,
                16,
                0,
                RV64_REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        ));
        let lifted = ext.try_lift(&inst, 0x100).unwrap();
        let LiftedInstr::Body(InstrAt { instr, .. }) = lifted else {
            panic!("expected public-values store body instruction");
        };

        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);
        assert_eq!(
            ctx.lines[0],
            format!("if (unlikely(!openvm_reveal(r1, r2, {width}u))) {{")
        );
    }

    #[test]
    fn rv64io_rejects_public_values_store_with_non_register_d() {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            Rv64LoadStoreOpcode::STORED.global_opcode(),
            [
                8,
                16,
                0,
                PUBLIC_VALUES_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        ));

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test]
    fn rv64io_ignores_non_store_public_values_shaped_instruction() {
        let ext = Rv64IoExtension::new(None).unwrap();
        let inst = RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            SystemOpcode::TERMINATE.global_opcode(),
            [
                8,
                16,
                0,
                RV64_REGISTER_AS as usize,
                PUBLIC_VALUES_AS as usize,
                1,
                0,
            ],
        ));

        assert!(ext.try_lift(&inst, 0x100).is_none());
    }

    #[test]
    fn reveal_accounts_for_the_offset_public_values_page() {
        let mut ctx = TestEmitCtx::default();
        RevealInstr {
            src_reg: Reg::new(5),
            ptr_reg: Reg::new(10),
            offset: 12,
            width: MemWidth::Word,
        }
        .emit_c(&mut ctx);

        assert_eq!(
            ctx.lines[0],
            "if (unlikely(!openvm_reveal(r5, (r10 + 0x0000000cull), 4u))) {"
        );
        assert_eq!(
            ctx.lines[3],
            format!("trace_page_access(state, (r10 + 0x0000000cull), 4u, {PUBLIC_VALUES_AS}u);")
        );
    }

    #[test]
    fn hint_storew_emits_the_three_slot_schedule() {
        let instr = HintStoreWInstr {
            ptr_reg: Reg::new(5),
        };
        assert!(instr.supports_preflight());

        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);

        assert_eq!(
            ctx.lines,
            [
                "if (unlikely(!openvm_hint_prepare(r5, 1u))) {",
                "trap;",
                "}",
                "reserve_preflight_writes(1u, 2u);",
                "uint64_t hint_word;",
                "openvm_hint_read_words(&hint_word, 1u);",
                "advance_timestamp(1);",
                "write_aligned_mem_block(r5, hint_word);",
            ]
        );
    }

    #[test]
    fn hint_buffer_emits_validation_reservation_and_three_slots_per_word() {
        let instr = HintBufferInstr {
            ptr_reg: Reg::new(5),
            num_words_reg: Reg::new(6),
            chip_idx: None,
        };
        assert!(instr.supports_preflight());

        let mut ctx = TestEmitCtx::default();
        instr.emit_c(&mut ctx);

        let emitted = ctx.lines.join("\n");
        assert!(emitted.contains("if (unlikely((r6 - 1ull) >= 1023ull)) {"));
        assert!(emitted.contains("if (unlikely(!openvm_hint_prepare(r5, (uint32_t)(r6)))) {"));
        assert!(emitted
            .contains("reserve_preflight_writes((uint32_t)(r6), ((uint32_t)(r6) * 3u - 2u));"));
        assert!(emitted.contains("uint64_t hint_words[1023u];"));
        assert!(emitted.contains("openvm_hint_read_words(hint_words, (uint32_t)(r6));"));
        assert!(emitted.contains("write_aligned_mem_block(r5, hint_words[0]);"));
        assert!(emitted.contains("advance_timestamp(2);"));
        assert!(emitted.contains(
            "write_aligned_mem_block((r5 + (uint64_t)hint_idx * 8ull), hint_words[hint_idx]);"
        ));
        assert!(!emitted.contains("trace_page_access"));
    }

    #[test_case(0x1122_3344, 6, 4, &[0x44, 0x33, 0x22, 0x11]; "word")]
    #[test_case(0x1122_3344_5566_7788, 3, 2, &[0x88, 0x77]; "halfword")]
    fn host_reveal_writes_requested_width(src_val: u64, addr: u64, width: u8, expected: &[u8]) {
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
            deferrals: &mut deferrals,
        };

        assert!(host_reveal(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            src_val,
            addr,
            width,
        ));

        let start = addr as usize;
        let end = start + usize::from(width);
        assert_eq!(&io.public_values[start..end], expected);
        assert!(io.public_values[..start].iter().all(|&byte| byte == 0));
        assert!(io.public_values[end..].iter().all(|&byte| byte == 0));
    }

    #[test_case(u64::MAX, 8; "address_overflow")]
    #[test_case(15, 2; "out_of_bounds")]
    #[test_case(0, 3; "invalid_width")]
    fn host_reveal_rejects_invalid_range(addr: u64, width: u8) {
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
            deferrals: &mut deferrals,
        };

        assert!(!host_reveal(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            u64::MAX,
            addr,
            width,
        ));
        assert!(io.public_values.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn host_hint_callbacks_validate_then_consume_without_touching_guest_memory() {
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
        hint_stream.set_hint(vec![1, 2, 3, 4, 5, 6, 7]);
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
            deferrals: &mut deferrals,
        };
        let ctx = &mut io as *mut OpenVmIoState<'_> as *mut c_void;

        assert!(!host_hint_prepare(ctx, 0, 1));
        assert!(!host_hint_prepare(ctx, 1, 0));
        assert!(!host_hint_prepare(ctx, 1, 1));
        assert!(!host_hint_prepare(
            ctx,
            0,
            (MAX_HINT_BUFFER_DWORDS + 1) as u32
        ));
        assert!(!host_hint_prepare(ctx, u64::MAX - 7, 1));
        assert_eq!(io.hint_stream.remaining(), 7);
        let mut hint = [0; 7];
        io.hint_stream.copy_to_slice(&mut hint);
        assert_eq!(hint, [1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(memory, original_memory);
    }

    #[test]
    fn host_hint_prepare_accepts_the_maximum_word_count() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let hint = (0..MAX_HINT_BUFFER_DWORDS * RV64_REGISTER_NUM_LIMBS)
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
