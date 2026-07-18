//! rvr lifter for the RISC-V I/O sub-extension: HINT_STORED, HINT_BUFFER,
//! and stores to public values, including REVEAL.
//!
//! TODO: check if other RISC-V instructions/opcodes can be separated into
//! extensions.
#![cfg(feature = "rvr")]

use std::{ffi::c_void, io::Write, iter::repeat_with};

use openvm_circuit::arch::rvr::io::{check_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode, PUBLIC_VALUES_AS,
};
use openvm_riscv_transpiler::{Rv64HintStoreOpcode, Rv64LoadStoreOpcode, Rv64Phantom};
use rand::Rng;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, MemWidth, Reg};
use rvr_openvm_lift::{
    air_index_to_c, decode_imm_cg, decode_reg, opcode_air_idx, AirIndex, ExtensionError,
    RvrExtension, RvrExtensionCtx, RvrInstruction, RvrRuntimeExtension,
};

/// HINT_STORED: pop one register word (8 bytes) from the hint stream into `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintStoreWInstr {
    pub ptr_reg: Reg,
}

impl ExtInstr for HintStoreWInstr {
    fn opname(&self) -> &str {
        "hint_storew"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg(self.ptr_reg);
        ctx.trace_mem_access(&ptr, RV64_MEMORY_AS);
        ctx.extern_call_without_page_flush("openvm_hint_storew", &[&ptr]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_BUFFER: pop `8 * reg[num_words_reg]` bytes from the hint stream and
/// write them sequentially starting at `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintBufferInstr {
    pub ptr_reg: Reg,
    pub num_words_reg: Reg,
    pub chip_idx: Option<AirIndex>,
}

impl ExtInstr for HintBufferInstr {
    fn opname(&self) -> &str {
        "hint_buffer"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg(self.ptr_reg);
        let n = ctx.read_reg(self.num_words_reg);
        // Block-entry already credits a static +1; emit the runtime
        // `(n - 1)` correction only when there is more than one row.
        let chip_idx = air_index_to_c(self.chip_idx);
        ctx.write_line(&format!("if ({n} > 1) {{"));
        ctx.trace_chip(chip_idx, &format!("{n} - 1"));
        ctx.write_line("}");
        ctx.write_line(&format!("if ({n} > 0) {{"));
        ctx.trace_mem_access_u64_range(&ptr, &n, RV64_MEMORY_AS);
        ctx.write_line("}");
        ctx.extern_call_without_page_flush("openvm_hint_buffer", &[&ptr, &n]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// Store bytes from `reg[src_reg]` into public values at `reg[ptr_reg] + offset`.
/// This is also used for REVEAL.
#[derive(Debug, Clone)]
pub struct RevealInstr {
    pub src_reg: Reg,
    pub ptr_reg: Reg,
    pub offset: u32,
    pub width: MemWidth,
}

impl ExtInstr for RevealInstr {
    fn opname(&self) -> &str {
        "reveal"
    }

    fn accesses_memory(&self) -> bool {
        false
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let src = ctx.read_reg(self.src_reg);
        let ptr = ctx.read_reg(self.ptr_reg);
        let addr = if self.offset == 0 {
            ptr.clone()
        } else {
            format!("({ptr} + 0x{:08x}u)", self.offset)
        };
        ctx.trace_mem_access(&addr, PUBLIC_VALUES_AS);
        let offset = format!("0x{:08x}u", self.offset);
        let width = self.width.bytes().to_string();
        ctx.extern_call_without_page_flush("openvm_reveal", &[&src, &ptr, &offset, &width]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_INPUT phantom: pop the next input record into the hint stream.
#[derive(Debug, Clone)]
pub struct HintInputInstr;

impl ExtInstr for HintInputInstr {
    fn opname(&self) -> &str {
        "hint_input"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.extern_call_without_page_flush("openvm_hint_input", &[]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// PRINT_STR phantom: print a UTF-8 string from guest memory to host stdout.
#[derive(Debug, Clone)]
pub struct PrintStrInstr {
    pub ptr_reg: Reg,
    pub len_reg: Reg,
}

impl ExtInstr for PrintStrInstr {
    fn opname(&self) -> &str {
        "print_str"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg_raw(self.ptr_reg);
        let len = ctx.read_reg_raw(self.len_reg);
        ctx.extern_call_without_page_flush("openvm_print_str", &[&ptr, &len]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_RANDOM phantom: fill the hint stream with `reg[num_words_reg] * 8`
/// random bytes drawn from the host's persistent RNG.
#[derive(Debug, Clone)]
pub struct HintRandomInstr {
    pub num_words_reg: Reg,
}

impl ExtInstr for HintRandomInstr {
    fn opname(&self) -> &str {
        "hint_random"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let n = ctx.read_reg_raw(self.num_words_reg);
        ctx.extern_call_without_page_flush("openvm_hint_random", &[&n]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// rvr extension for the RISC-V I/O instructions HINT_STORED, HINT_BUFFER, and
/// stores to public values, including REVEAL.
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
                instr: Instr::Ext(Box::new(HintStoreWInstr { ptr_reg })),
                source_loc: None,
            }));
        }

        if opcode == Rv64HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
            let num_words_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(HintBufferInstr {
                    ptr_reg,
                    num_words_reg,
                    chip_idx: self.hint_store_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if let Some(width) = public_values_store_width(insn) {
            let src_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            let offset = decode_imm_cg(insn);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(RevealInstr {
                    src_reg,
                    ptr_reg,
                    offset,
                    width,
                })),
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
}

pub struct Rv64IoRuntimeHooks;

impl RvrRuntimeExtension for Rv64IoRuntimeHooks {
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        let register_fn: RegisterRv64IoFn = unsafe {
            let sym = lib
                .get::<RegisterRv64IoFn>(b"register_rv64io_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        let callbacks = Rv64IoHostCallbacks {
            hint_storew: host_hint_storew,
            hint_buffer: host_hint_buffer,
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

/// rvr extension for the RISC-V base IO phantoms HINT_INPUT, PRINT_STR,
/// HINT_RANDOM, and the extension hint-stream setter.
pub struct Rv64IExtension;

impl Rv64IExtension {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Rv64IExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for Rv64IExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        if insn.opcode.as_usize() != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }
        let discriminant = (insn.c & 0xffff) as u16;
        let phantom = Rv64Phantom::from_repr(discriminant)?;
        let instr: Box<dyn ExtInstr> = match phantom {
            Rv64Phantom::HintInput => Box::new(HintInputInstr),
            Rv64Phantom::PrintStr => Box::new(PrintStrInstr {
                ptr_reg: decode_reg(insn.a),
                len_reg: decode_reg(insn.b),
            }),
            Rv64Phantom::HintRandom => Box::new(HintRandomInstr {
                num_words_reg: decode_reg(insn.a),
            }),
        };
        Some(LiftedInstr::Body(InstrAt {
            pc,
            instr: Instr::Ext(instr),
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64i_phantom_callbacks.h",
            include_str!("../c/rv64i_phantom_callbacks.h"),
        )]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64i_phantom_callbacks.c",
            include_str!("../c/rv64i_phantom_callbacks.c"),
        )]
    }
}

pub struct Rv64IRuntimeHooks;

impl RvrRuntimeExtension for Rv64IRuntimeHooks {
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        let register_fn: RegisterRv64IPhantomFn = unsafe {
            let sym = lib
                .get::<RegisterRv64IPhantomFn>(b"register_rv64i_phantom_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        let callbacks = Rv64IPhantomCallbacks {
            hint_input: host_hint_input,
            print_str: host_print_str,
            hint_random: host_hint_random,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

type RegisterRv64IPhantomFn = unsafe extern "C" fn(*const Rv64IPhantomCallbacks);
type RegisterRv64IoFn = unsafe extern "C" fn(*const Rv64IoHostCallbacks);

/// Must match the C `Rv64IPhantomCallbacks` layout in `rv64i_phantom_callbacks.c`.
#[repr(C)]
pub struct Rv64IPhantomCallbacks {
    pub hint_input: extern "C" fn(*mut c_void),
    pub print_str: extern "C" fn(*mut c_void, u64, u32),
    pub hint_random: extern "C" fn(*mut c_void, u32),
}

/// Must match the C `Rv64IoHostCallbacks` layout in `rv64io_callbacks.c`.
#[repr(C)]
pub struct Rv64IoHostCallbacks {
    pub hint_storew: extern "C" fn(*mut c_void, u64),
    pub hint_buffer: extern "C" fn(*mut c_void, u64, u32),
    pub reveal: extern "C" fn(*mut c_void, u64, u64, u32, u32),
}

// ── Callback implementations ────────────────────────────────────────────────

/// Makes the next input record available without copying its payload.
pub extern "C" fn host_hint_input(ctx: *mut c_void) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if let Some(bytes) = io.input_stream.pop_front() {
        io.hint_stream.set_input(bytes);
    } else {
        io.hint_stream.clear();
    }
}

/// PrintStr: read UTF-8 from guest memory and print to stdout.
pub extern "C" fn host_print_str(ctx: *mut c_void, ptr: u64, len: u32) {
    let io = unsafe { &*(ctx as *const OpenVmIoState<'_>) };
    if len > 0 && !io.memory_ptr.is_null() {
        check_mem_bounds_range(ptr, len as usize);
        let slice =
            unsafe { std::slice::from_raw_parts(io.memory_ptr.add(ptr as usize), len as usize) };
        let _ = std::io::stdout().write_all(slice);
        let _ = std::io::stdout().flush();
    }
}

/// HintRandom: refill the hint stream with `num_words * RV64_REGISTER_NUM_LIMBS` random
/// bytes drawn from VmState's persistent RNG.
pub extern "C" fn host_hint_random(ctx: *mut c_void, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let nbytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    io.hint_stream
        .set_hint_from_iter(repeat_with(|| io.rng.random::<u8>()).take(nbytes));
}

/// HINT_STOREW: pop one rv64 register-width word (8 bytes) from the hint stream
/// and write it to guest memory at `dest_addr`.
pub extern "C" fn host_hint_storew(ctx: *mut c_void, dest_addr: u64) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if io.hint_stream.remaining() < RV64_REGISTER_NUM_LIMBS || io.memory_ptr.is_null() {
        return;
    }
    check_mem_bounds_range(dest_addr, RV64_REGISTER_NUM_LIMBS);
    let dst = unsafe {
        std::slice::from_raw_parts_mut(
            io.memory_ptr.add(dest_addr as usize),
            RV64_REGISTER_NUM_LIMBS,
        )
    };
    io.hint_stream.copy_to_slice(dst);
}

/// HINT_BUFFER: pop `num_words * RV64_REGISTER_NUM_LIMBS` bytes from the hint stream
/// and copy them into guest memory.
pub extern "C" fn host_hint_buffer(ctx: *mut c_void, dest_addr: u64, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let nbytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    if io.hint_stream.remaining() < nbytes || io.memory_ptr.is_null() {
        return;
    }
    check_mem_bounds_range(dest_addr, nbytes);
    let dst =
        unsafe { std::slice::from_raw_parts_mut(io.memory_ptr.add(dest_addr as usize), nbytes) };
    io.hint_stream.copy_to_slice(dst);
}

/// Host callback for stores to `PUBLIC_VALUES_AS`.
/// Cost corrections are handled in generated C.
pub extern "C" fn host_reveal(ctx: *mut c_void, src_val: u64, ptr: u64, offset: u32, width: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let start = ptr as usize + offset as usize;
    let width = width as usize;
    let end = start + width;
    assert!(
        end <= io.public_values.len(),
        "reveal out of bounds: writing bytes [{start}..{end}) but public_values size is {} (configured via SystemConfig::with_public_values_bytes or SystemConfig::with_public_values)",
        io.public_values.len(),
    );
    io.public_values[start..end].copy_from_slice(&src_val.to_le_bytes()[..width]);
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, ptr::null_mut};

    use openvm_circuit::arch::HintStream;
    use openvm_instructions::instruction::Instruction;
    use p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};

    use super::*;

    #[derive(Default)]
    struct TestEmitCtx {
        lines: Vec<String>,
    }

    impl ExtEmitCtx for TestEmitCtx {
        fn read_reg(&mut self, idx: u8) -> String {
            format!("r{idx}")
        }

        fn read_reg_raw(&mut self, idx: u8) -> String {
            format!("r{idx}")
        }

        fn write_reg(&mut self, _idx: u8, _val: &str) {}

        fn write_reg_raw(&mut self, _idx: u8, _val: &str) {}

        fn write_line(&mut self, s: &str) {
            self.lines.push(s.to_string());
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

        fn extern_call(&mut self, name: &str, args: &[&str]) {
            self.write_line(&format!("{name}({});", args.join(", ")));
        }

        fn extern_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
            let tmp = format!("tmp{}", self.lines.len());
            self.write_line(&format!("{ret_ty} {tmp} = {name}({});", args.join(", ")));
            tmp
        }

        fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
            self.write_line(&format!("trace_chip(state, {chip_idx}u, {count_expr});"));
        }

        fn trace_mem_access(&mut self, addr: &str, addr_space: u32) {
            self.write_line(&format!("trace_mem_access(state, {addr}, {addr_space}u);"));
        }

        fn trace_mem_access_u64_range(
            &mut self,
            base_addr: &str,
            num_dwords: &str,
            addr_space: u32,
        ) {
            self.write_line(&format!(
                "trace_mem_access_u64_range(state, {base_addr}, {num_dwords}, {addr_space}u);"
            ));
        }
    }

    #[test]
    fn rv64io_lifts_public_values_store_widths() {
        let ext = Rv64IoExtension::new(None).unwrap();

        for (opcode, width) in [
            (Rv64LoadStoreOpcode::STORED, 8),
            (Rv64LoadStoreOpcode::STOREW, 4),
            (Rv64LoadStoreOpcode::STOREH, 2),
            (Rv64LoadStoreOpcode::STOREB, 1),
        ] {
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
            let LiftedInstr::Body(InstrAt {
                instr: Instr::Ext(instr),
                ..
            }) = lifted
            else {
                panic!("expected public-values store extension instruction");
            };

            let mut ctx = TestEmitCtx::default();
            instr.emit_c(&mut ctx);
            assert_eq!(
                ctx.lines[1],
                format!("openvm_reveal(r1, r2, 0x00000000u, {width});")
            );
        }
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
    fn reveal_traces_the_offset_public_values_address() {
        let mut ctx = TestEmitCtx::default();
        RevealInstr {
            src_reg: 5,
            ptr_reg: 10,
            offset: 12,
            width: MemWidth::Word,
        }
        .emit_c(&mut ctx);

        assert_eq!(
            ctx.lines[0],
            format!("trace_mem_access(state, (r10 + 0x0000000cu), {PUBLIC_VALUES_AS}u);")
        );
        assert_eq!(ctx.lines[1], "openvm_reveal(r5, r10, 0x0000000cu, 4);");
    }

    #[test]
    fn host_reveal_writes_public_values_slice() {
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

        host_reveal(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            0x11223344,
            4,
            2,
            4,
        );

        assert_eq!(&io.public_values[6..10], &[0x44, 0x33, 0x22, 0x11]);
    }

    #[test]
    fn host_reveal_honors_store_width() {
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

        host_reveal(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            0x1122334455667788,
            3,
            0,
            2,
        );

        assert_eq!(&io.public_values[3..5], &[0x88, 0x77]);
        assert_eq!(io.public_values[5], 0);
    }

    #[test]
    fn hint_buffer_traces_u64_range() {
        let mut ctx = TestEmitCtx::default();
        HintBufferInstr {
            ptr_reg: 1,
            num_words_reg: 2,
            chip_idx: None,
        }
        .emit_c(&mut ctx);

        assert!(
            ctx.lines
                .iter()
                .any(|l| l.contains("trace_mem_access_u64_range")),
            "expected trace_mem_access_u64_range, got: {:#?}",
            ctx.lines
        );
    }

    #[test]
    fn host_hint_buffer_copies_to_guest_memory() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        hint_stream.set_hint((10u8..22).collect());

        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 16];
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

        host_hint_buffer(&mut io as *mut OpenVmIoState<'_> as *mut c_void, 3, 1);

        assert_eq!(&memory[3..11], &(10u8..18).collect::<Vec<_>>());
        assert_eq!(io.hint_stream.remaining(), 4);
    }

    #[test]
    fn host_input_callbacks_expose_length_payload_and_padding() {
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

        host_hint_input(ctx);
        host_hint_storew(ctx, 0);
        host_hint_buffer(ctx, 8, 2);

        assert_eq!(&memory[..8], &(payload.len() as u64).to_le_bytes());
        assert_eq!(&memory[8..17], payload);
        assert_eq!(&memory[17..], &[0; 7]);
        assert_eq!(io.hint_stream.remaining(), 0);
        assert!(io.input_stream.is_empty());
    }
}
