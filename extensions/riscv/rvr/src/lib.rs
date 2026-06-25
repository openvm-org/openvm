//! rvr lifter for the RISC-V I/O sub-extension (HINT_STORED, HINT_BUFFER,
//! REVEAL).
//!
//! TODO: check if other RISC-V instructions/opcodes can be separated into
//! extensions.
#![cfg(feature = "rvr")]

use std::{ffi::c_void, io::Write};

use openvm_circuit::arch::rvr::io::{check_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{
    instruction::Instruction,
    riscv::{RV64_MEMORY_AS, RV64_REGISTER_NUM_LIMBS},
    LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{Rv64HintStoreOpcode, Rv64LoadStoreOpcode, Rv64Phantom};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::Rng;
use rvr_openvm_ext_ffi_common::AS_PUBLIC_VALUES;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_to_c, decode_imm_cg, decode_reg, opcode_air_idx, AirIndex, ExtensionError,
    RvrExtension, RvrExtensionCtx,
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
        ctx.extern_call("openvm_hint_storew", &[&ptr]);
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
        ctx.extern_call("openvm_hint_buffer", &[&ptr, &n]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// REVEAL: write `reg[src_reg]` to user public-output
/// address space at `reg[ptr_reg] + offset`.
#[derive(Debug, Clone)]
pub struct RevealInstr {
    pub src_reg: Reg,
    pub ptr_reg: Reg,
    pub offset: u32,
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
        ctx.trace_mem_access(&addr, AS_PUBLIC_VALUES);
        let offset = format!("0x{:08x}u", self.offset);
        ctx.extern_call("openvm_reveal", &[&src, &ptr, &offset]);
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
        ctx.extern_call("openvm_hint_input", &[]);
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
        ctx.extern_call("openvm_print_str", &[&ptr, &len]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_RANDOM phantom: fill the hint stream with `reg[num_words_reg] * 4`
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
        ctx.extern_call("openvm_hint_random", &[&n]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// rvr extension for the RISC-V I/O instructions HINT_STORED, HINT_BUFFER, and
/// REVEAL.
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

impl<F: PrimeField32> RvrExtension<F> for Rv64IoExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
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

        // REVEAL: STORED with address-space e = AS_PUBLIC_VALUES.
        if opcode == Rv64LoadStoreOpcode::STORED.global_opcode_usize()
            && insn.e.as_canonical_u32() == AS_PUBLIC_VALUES
        {
            let src_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            let offset = decode_imm_cg(insn);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(RevealInstr {
                    src_reg,
                    ptr_reg,
                    offset,
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
            hint_storew: host_hint_storew::<F>,
            hint_buffer: host_hint_buffer::<F>,
            reveal: host_reveal::<F>,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
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

impl<F: PrimeField32> RvrExtension<F> for Rv64IExtension {
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        if insn.opcode.as_usize() != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }
        let discriminant = (insn.c.as_canonical_u32() & 0xffff) as u16;
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
            hint_input: host_hint_input::<F>,
            print_str: host_print_str::<F>,
            hint_random: host_hint_random::<F>,
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
    pub print_str: extern "C" fn(*mut c_void, u32, u32),
    pub hint_random: extern "C" fn(*mut c_void, u32),
}

/// Must match the C `Rv64IoHostCallbacks` layout in `rv64io_callbacks.c`.
#[repr(C)]
pub struct Rv64IoHostCallbacks {
    pub hint_storew: extern "C" fn(*mut c_void, u32),
    pub hint_buffer: extern "C" fn(*mut c_void, u32, u32),
    pub reveal: extern "C" fn(*mut c_void, u64, u32, u32),
}

// ── Callback implementations ────────────────────────────────────────────────

/// HintInput: pop next input record from VmState's input_stream and overwrite
/// the active hint stream with `[len: u64 LE][data][padding to 8-byte align]`,
/// each byte stored as one field element.
pub extern "C" fn host_hint_input<F: PrimeField32>(ctx: *mut c_void) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    io.hint_stream.clear();
    if let Some(mut vec) = io.input_stream.pop_front() {
        let data_len = vec.len();
        let len_bytes = (data_len as u64).to_le_bytes();
        for &b in &len_bytes {
            io.hint_stream.push_back(F::from_u8(b));
        }
        let padded_len = data_len.div_ceil(RV64_REGISTER_NUM_LIMBS) * RV64_REGISTER_NUM_LIMBS;
        vec.resize(padded_len, F::ZERO);
        io.hint_stream.extend(vec);
    }
}

/// PrintStr: read UTF-8 from guest memory and print to stdout.
pub extern "C" fn host_print_str<F: PrimeField32>(ctx: *mut c_void, ptr: u32, len: u32) {
    let io = unsafe { &*(ctx as *const OpenVmIoState<'_, F>) };
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
pub extern "C" fn host_hint_random<F: PrimeField32>(ctx: *mut c_void, num_words: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let nbytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    io.hint_stream.clear();
    for _ in 0..nbytes {
        io.hint_stream.push_back(F::from_u8(io.rng.random::<u8>()));
    }
}

/// HINT_STOREW: pop one rv64 register-width word (8 bytes) from the hint stream
/// and write it to guest memory at `dest_addr`.
pub extern "C" fn host_hint_storew<F: PrimeField32>(ctx: *mut c_void, dest_addr: u32) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    if io.hint_stream.len() < RV64_REGISTER_NUM_LIMBS || io.memory_ptr.is_null() {
        return;
    }
    check_mem_bounds_range(dest_addr, RV64_REGISTER_NUM_LIMBS);
    let mut bytes = [0u8; RV64_REGISTER_NUM_LIMBS];
    for byte in &mut bytes {
        *byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            io.memory_ptr.add(dest_addr as usize),
            bytes.len(),
        );
    }
}

/// HINT_BUFFER: pop `num_words * RV64_REGISTER_NUM_LIMBS` field elements from the hint stream
/// and copy them as bytes into guest memory.
pub extern "C" fn host_hint_buffer<F: PrimeField32>(
    ctx: *mut c_void,
    dest_addr: u32,
    num_words: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let nbytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    if io.hint_stream.len() < nbytes || io.memory_ptr.is_null() {
        return;
    }
    check_mem_bounds_range(dest_addr, nbytes);
    let dst = unsafe { io.memory_ptr.add(dest_addr as usize) };
    for i in 0..nbytes {
        let byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
        unsafe { *dst.add(i) = byte };
    }
}

/// REVEAL: write public output bytes directly into the guest's `PUBLIC_VALUES_AS`
/// byte slice. Cost corrections handled in C.
pub extern "C" fn host_reveal<F: PrimeField32>(
    ctx: *mut c_void,
    src_val: u64,
    ptr: u32,
    offset: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let start = ptr as usize + offset as usize;
    let end = start + RV64_REGISTER_NUM_LIMBS;
    assert!(
        end <= io.public_values.len(),
        "reveal out of bounds: writing bytes [{start}..{end}) but public_values size is {} (configured via SystemConfig::with_public_values_bytes or SystemConfig::with_public_values)",
        io.public_values.len(),
    );
    io.public_values[start..end].copy_from_slice(&src_val.to_le_bytes());
}


#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

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
    fn reveal_traces_the_offset_public_values_address() {
        let mut ctx = TestEmitCtx::default();
        RevealInstr {
            src_reg: 5,
            ptr_reg: 10,
            offset: 12,
        }
        .emit_c(&mut ctx);

        assert_eq!(
            ctx.lines[0],
            format!("trace_mem_access(state, (r10 + 0x0000000cu), {AS_PUBLIC_VALUES}u);")
        );
        assert_eq!(ctx.lines[1], "openvm_reveal(r5, r10, 0x0000000cu);");
    }

    #[test]
    fn host_reveal_writes_public_values_slice() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = VecDeque::new();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0u8; 16];
        let mut public_values = vec![0u8; 16];
        let mut deferrals = Vec::new();

        let mut io = OpenVmIoState::<BabyBear> {
            input_stream: &mut input_stream,
            hint_stream: &mut hint_stream,
            rng: &mut rng,
            memory_ptr: memory.as_mut_ptr(),
            public_values: &mut public_values,
            deferral_memory: std::ptr::null_mut(),
            deferral_memory_len: 0,
            deferrals: &mut deferrals,
        };

        host_reveal::<BabyBear>(
            &mut io as *mut OpenVmIoState<'_, BabyBear> as *mut c_void,
            0x11223344,
            4,
            2,
        );

        assert_eq!(&io.public_values[6..10], &[0x44, 0x33, 0x22, 0x11]);
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
}
