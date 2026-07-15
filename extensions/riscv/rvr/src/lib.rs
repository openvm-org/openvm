//! rvr lifter for the RISC-V I/O sub-extension (HINT_STORED, HINT_BUFFER,
//! REVEAL).
//!
//! TODO: check if other RISC-V instructions/opcodes can be separated into
//! extensions.
#![cfg(feature = "rvr")]

use std::{ffi::c_void, io::Write};

use openvm_circuit::arch::rvr::io::{check_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{
    instruction::Instruction, riscv::RV64_REGISTER_NUM_LIMBS, LocalOpcode, SystemOpcode,
};
use openvm_riscv_transpiler::{Rv64HintStoreOpcode, Rv64LoadStoreOpcode, Rv64Phantom};
use openvm_stark_backend::p3_field::PrimeField32;
use rand::Rng;
use rvr_openvm_ext_ffi_common::AS_PUBLIC_VALUES;
use rvr_openvm_ir::{ExtEmitCtx, ExtInstr, InlineRecordShape, Instr, InstrAt, LiftedInstr, Reg};
use rvr_openvm_lift::{
    air_index_codegen_fingerprint, air_index_to_c, decode_imm_cg, decode_reg, opcode_air_idx,
    AirIndex, ExtensionError, RvrExtension, RvrExtensionCtx,
};

/// Byte geometry of the packed `Rv64HintStoreRecordHeader + N * Var`
/// consumer record. Circuit-side assertions pin these constants to the real
/// Rust types; the per-row sum is the conservative metered-height capacity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HintStoreRecordDescriptor {
    pub header_size: usize,
    pub header_align: usize,
    pub var_size: usize,
    pub var_align: usize,
    pub capacity_per_row: usize,
}

impl HintStoreRecordDescriptor {
    pub const fn new() -> Self {
        Self {
            header_size: 32,
            header_align: 4,
            var_size: 20,
            var_align: 4,
            capacity_per_row: 52,
        }
    }

    pub const fn record_size(self, rows: usize) -> usize {
        self.header_size + self.var_size * rows
    }
}

impl Default for HintStoreRecordDescriptor {
    fn default() -> Self {
        Self::new()
    }
}

/// HINT_STORED: pop one register word (8 bytes) from the hint stream into `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintStoreWInstr {
    pub from_pc: u32,
    pub ptr_reg: Reg,
    pub chip_idx: Option<AirIndex>,
}

impl ExtInstr for HintStoreWInstr {
    fn opname(&self) -> &str {
        "hint_storew"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let (ptr, from_timestamp, mem_ptr_prev_timestamp) = ctx.read_reg_with_trace(self.ptr_reg);
        // HINT_STORED has the same 3-tick AIR shape as HINT_BUFFER:
        // mem_ptr read, a num_words placeholder tick, then the memory write.
        ctx.trace_timestamp();
        let chip_idx = if ctx.inline_record_enabled() {
            air_index_to_c(self.chip_idx)
        } else {
            u32::MAX
        };
        ctx.extern_call(
            "openvm_hint_storew",
            &[
                "state",
                &ptr,
                &format!("{}u", self.from_pc),
                &from_timestamp,
                &format!(
                    "{}u",
                    u32::from(self.ptr_reg) * RV64_REGISTER_NUM_LIMBS as u32
                ),
                &mem_ptr_prev_timestamp,
                &format!("{chip_idx}u"),
            ],
        );
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::CustomVariableRows {
            capacity_per_row: HintStoreRecordDescriptor::new().capacity_per_row,
        })
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_BUFFER: pop `8 * reg[num_words_reg]` bytes from the hint stream and
/// write them sequentially starting at `mem[reg[ptr_reg]]`.
#[derive(Debug, Clone)]
pub struct HintBufferInstr {
    pub from_pc: u32,
    pub ptr_reg: Reg,
    pub num_words_reg: Reg,
    pub chip_idx: Option<AirIndex>,
}

impl ExtInstr for HintBufferInstr {
    fn opname(&self) -> &str {
        "hint_buffer"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let (ptr, from_timestamp, mem_ptr_prev_timestamp) = ctx.read_reg_with_trace(self.ptr_reg);
        let (n, _, num_words_prev_timestamp) = ctx.read_reg_with_trace(self.num_words_reg);
        // Block-entry already credits a static +1; emit the runtime
        // `(n - 1)` correction only when there is more than one row.
        let chip_idx = air_index_to_c(self.chip_idx);
        ctx.write_line(&format!("if ({n} > 1) {{"));
        ctx.trace_chip(chip_idx, &format!("{n} - 1"));
        ctx.write_line("}");
        ctx.write_line(&format!("if ({n} > 0) {{"));
        let direct_chip_idx = if ctx.inline_record_enabled() {
            chip_idx
        } else {
            u32::MAX
        };
        ctx.extern_call(
            "openvm_hint_buffer",
            &[
                "state",
                &ptr,
                &n,
                &format!("{}u", self.from_pc),
                &from_timestamp,
                &format!(
                    "{}u",
                    u32::from(self.ptr_reg) * RV64_REGISTER_NUM_LIMBS as u32
                ),
                &mem_ptr_prev_timestamp,
                &format!(
                    "{}u",
                    u32::from(self.num_words_reg) * RV64_REGISTER_NUM_LIMBS as u32
                ),
                &num_words_prev_timestamp,
                &format!("{direct_chip_idx}u"),
            ],
        );
        ctx.write_line("}");
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::CustomVariableRows {
            capacity_per_row: HintStoreRecordDescriptor::new().capacity_per_row,
        })
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
        // Interpreter (LoadStore chip) access order for a store: rs1 (ptr)
        // register read, then rs2 (src) register read, then the public-values
        // write. Preflight emits the LoadStore row inline; other modes use
        // the verbose trace default.
        let (src, ptr) = ctx.trace_store_u64_as(
            self.src_reg,
            self.ptr_reg,
            self.offset,
            AS_PUBLIC_VALUES,
            4, // Rv64LoadStoreOpcode::STORED ordinal
        );
        let offset = format!("0x{:08x}u", self.offset);
        ctx.extern_call("openvm_reveal", &[&src, &ptr, &offset]);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Alu3)
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

/// HINT_INPUT phantom: pop the next input record into the hint stream.
#[derive(Debug, Clone)]
pub struct HintInputInstr {
    pub operands: [u32; 3],
}

impl ExtInstr for HintInputInstr {
    fn opname(&self) -> &str {
        "hint_input"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.extern_call("openvm_hint_input", &[]);
        ctx.trace_phantom_record(self.operands);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
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
    pub operands: [u32; 3],
}

impl ExtInstr for PrintStrInstr {
    fn opname(&self) -> &str {
        "print_str"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.read_reg_raw(self.ptr_reg);
        let len = ctx.read_reg_raw(self.len_reg);
        ctx.extern_call("openvm_print_str", &[&ptr, &len]);
        ctx.trace_phantom_record(self.operands);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
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
    pub operands: [u32; 3],
}

impl ExtInstr for HintRandomInstr {
    fn opname(&self) -> &str {
        "hint_random"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let n = ctx.read_reg_raw(self.num_words_reg);
        ctx.extern_call("openvm_hint_random", &[&n]);
        ctx.trace_phantom_record(self.operands);
    }

    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        Some(InlineRecordShape::Custom { record_size: 20 })
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
        let hint_buffer_chip_idx = opcode_air_idx(ctx, Rv64HintStoreOpcode::HINT_BUFFER)?;
        if hint_store_chip_idx != hint_buffer_chip_idx {
            return Err(ExtensionError::SharedAirMismatch {
                first_opcode: Rv64HintStoreOpcode::HINT_STORED.global_opcode(),
                second_opcode: Rv64HintStoreOpcode::HINT_BUFFER.global_opcode(),
                first_air: hint_store_chip_idx,
                second_air: hint_buffer_chip_idx,
            });
        }
        Ok(Self {
            hint_store_chip_idx,
        })
    }
}

impl<F: PrimeField32> RvrExtension<F> for Rv64IoExtension {
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        Some(air_index_codegen_fingerprint(
            b"openvm-rv64io-rvr-v2",
            &[self.hint_store_chip_idx],
        ))
    }

    fn try_lift(&self, insn: &Instruction<F>, pc: u64) -> Option<LiftedInstr> {
        let opcode = insn.opcode.as_usize();

        if opcode == Rv64HintStoreOpcode::HINT_STORED.global_opcode_usize() {
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(HintStoreWInstr {
                    from_pc: pc as u32,
                    ptr_reg,
                    chip_idx: self.hint_store_chip_idx,
                })),
                source_loc: None,
            }));
        }

        if opcode == Rv64HintStoreOpcode::HINT_BUFFER.global_opcode_usize() {
            let num_words_reg = decode_reg(insn.a);
            let ptr_reg = decode_reg(insn.b);
            return Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Instr::Ext(Box::new(HintBufferInstr {
                    from_pc: pc as u32,
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

/// rvr extension for the RISC-V base IO phantoms HINT_INPUT, PRINT_STR, and HINT_RANDOM.
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
    fn codegen_fingerprint(&self) -> Option<Vec<u8>> {
        Some(b"openvm-rv64i-rvr-v2".to_vec())
    }

    fn try_lift(&self, insn: &Instruction<F>, pc: u64) -> Option<LiftedInstr> {
        if insn.opcode.as_usize() != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }
        let discriminant = (insn.c.as_canonical_u32() & 0xffff) as u16;
        let phantom = Rv64Phantom::from_repr(discriminant)?;
        let operands = [insn.a, insn.b, insn.c].map(|value| value.as_canonical_u32());
        let instr: Box<dyn ExtInstr> = match phantom {
            Rv64Phantom::HintInput => Box::new(HintInputInstr { operands }),
            Rv64Phantom::PrintStr => Box::new(PrintStrInstr {
                ptr_reg: decode_reg(insn.a),
                len_reg: decode_reg(insn.b),
                operands,
            }),
            Rv64Phantom::HintRandom => Box::new(HintRandomInstr {
                num_words_reg: decode_reg(insn.a),
                operands,
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
    pub print_str: extern "C" fn(*mut c_void, u64, u32),
    pub hint_random: extern "C" fn(*mut c_void, u32),
}

/// Must match the C `Rv64IoHostCallbacks` layout in `rv64io_callbacks.c`.
#[repr(C)]
pub struct Rv64IoHostCallbacks {
    pub hint_storew: extern "C" fn(*mut c_void, u64) -> u64,
    pub hint_buffer: extern "C" fn(*mut c_void, u64, u32, u32) -> u64,
    pub reveal: extern "C" fn(*mut c_void, u64, u64, u32),
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
pub extern "C" fn host_print_str<F: PrimeField32>(ctx: *mut c_void, ptr: u64, len: u32) {
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
///
/// Error-path note: exhausted hint streams currently return a zero word and
/// continue on the rvr host-callback path, while interpreter preflight returns
/// `ExecutionError::HintOutOfBounds`. Well-formed transpiled programs provision
/// enough hints; M4 documents this as an unreachable error-equivalence gap.
pub extern "C" fn host_hint_storew<F: PrimeField32>(ctx: *mut c_void, dest_addr: u64) -> u64 {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    if io.hint_stream.len() < RV64_REGISTER_NUM_LIMBS {
        return 0;
    }
    check_mem_bounds_range(dest_addr, RV64_REGISTER_NUM_LIMBS);
    let mut bytes = [0u8; RV64_REGISTER_NUM_LIMBS];
    for byte in &mut bytes {
        *byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
    }
    u64::from_le_bytes(bytes)
}

/// HINT_BUFFER: pop `num_words * RV64_REGISTER_NUM_LIMBS` field elements from the hint stream
/// and copy them as bytes into guest memory.
///
/// See `host_hint_storew` for the deliberate M4-documented exhausted-stream
/// error-path asymmetry with interpreter preflight.
pub extern "C" fn host_hint_buffer<F: PrimeField32>(
    ctx: *mut c_void,
    dest_addr: u64,
    num_words: u32,
    word_index: u32,
) -> u64 {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let nbytes = num_words as usize * RV64_REGISTER_NUM_LIMBS;
    if word_index == 0 {
        check_mem_bounds_range(dest_addr, nbytes);
        if io.hint_stream.len() < nbytes {
            return 0;
        }
    }
    if word_index >= num_words || io.hint_stream.len() < RV64_REGISTER_NUM_LIMBS {
        return 0;
    }
    let mut bytes = [0u8; RV64_REGISTER_NUM_LIMBS];
    for byte in &mut bytes {
        *byte = io.hint_stream.pop_front().unwrap().as_canonical_u32() as u8;
    }
    u64::from_le_bytes(bytes)
}

/// REVEAL: write public output bytes directly into the guest's `PUBLIC_VALUES_AS`
/// byte slice. Cost corrections handled in C.
pub extern "C" fn host_reveal<F: PrimeField32>(
    ctx: *mut c_void,
    src_val: u64,
    ptr: u64,
    offset: u32,
) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_, F>) };
    let Some((start, end)) = reveal_public_values_bounds(ptr, offset, io.public_values.len())
    else {
        panic!(
            "reveal out of bounds: ptr={ptr} offset={offset} write_size={} but public_values size is {} (configured via SystemConfig::with_public_values_bytes or SystemConfig::with_public_values)",
            RV64_REGISTER_NUM_LIMBS,
            io.public_values.len(),
        );
    };
    io.public_values[start..end].copy_from_slice(&src_val.to_le_bytes());
}

fn reveal_public_values_bounds(
    ptr: u64,
    offset: u32,
    public_values_len: usize,
) -> Option<(usize, usize)> {
    let start = ptr.checked_add(u64::from(offset))?;
    let end = start.checked_add(RV64_REGISTER_NUM_LIMBS as u64)?;
    if end > u64::try_from(public_values_len).ok()? {
        return None;
    }
    Some((usize::try_from(start).ok()?, usize::try_from(end).ok()?))
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
            self.lines.push(format!("trace_reg_read(state, {idx});"));
            format!("r{idx}")
        }

        fn read_reg_with_trace(&mut self, idx: u8) -> (String, String, String) {
            (self.read_reg(idx), "0u".to_string(), "0u".to_string())
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

        fn trace_wr_as_u64(&mut self, addr: &str, val: &str, addr_space: u32) {
            self.write_line(&format!(
                "trace_wr_as_u64(state, {addr}, {val}, {addr_space}u);"
            ));
        }

        fn trace_timestamp(&mut self) {
            self.write_line("trace_timestamp(state);");
        }
    }

    #[test]
    fn reveal_traces_interpreter_order_and_writes_public_values_address() {
        let mut ctx = TestEmitCtx::default();
        RevealInstr {
            src_reg: 5,
            ptr_reg: 10,
            offset: 12,
        }
        .emit_c(&mut ctx);

        // Interpreter store parity: rs1 (ptr) tick, rs2 (src) tick, then the
        // value-carrying public-values write.
        assert_eq!(ctx.lines[0], "trace_reg_read(state, 10);");
        assert_eq!(ctx.lines[1], "trace_reg_read(state, 5);");
        assert_eq!(
            ctx.lines[2],
            format!("trace_wr_as_u64(state, (r10 + 0x0000000cu), r5, {AS_PUBLIC_VALUES}u);")
        );
        assert_eq!(ctx.lines[3], "openvm_reveal(r5, r10, 0x0000000cu);");
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
    fn reveal_public_values_bounds_accepts_valid_range() {
        assert_eq!(reveal_public_values_bounds(4, 2, 16), Some((6, 14)));
    }

    #[test]
    fn reveal_public_values_bounds_rejects_64_bit_overflow() {
        assert_eq!(reveal_public_values_bounds(u64::MAX - 3, 8, 16), None);
    }

    #[test]
    fn reveal_public_values_bounds_rejects_out_of_range_write() {
        assert_eq!(reveal_public_values_bounds(9, 0, 16), None);
    }

    #[test]
    fn hint_buffer_uses_traced_host_callback() {
        let mut ctx = TestEmitCtx::default();
        HintBufferInstr {
            from_pc: 0x20,
            ptr_reg: 1,
            num_words_reg: 2,
            chip_idx: None,
        }
        .emit_c(&mut ctx);

        assert!(
            ctx.lines
                .iter()
                .any(|l| l.contains("openvm_hint_buffer(state, r1, r2,")),
            "expected stateful openvm_hint_buffer call, got: {:#?}",
            ctx.lines
        );
    }
}
