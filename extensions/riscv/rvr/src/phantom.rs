//! RV64-specific phantom instruction lifting and host callbacks.

use std::{ffi::c_void, io::Write, iter::repeat_with};

use openvm_circuit::arch::rvr::io::{checked_mem_bounds_range, OpenVmIoState};
use openvm_instructions::{riscv::RV64_REGISTER_BYTES, LocalOpcode, SystemOpcode};
use openvm_platform::memory::MEM_SIZE;
use openvm_riscv_transpiler::Rv64Phantom;
use rand::Rng;
use rvr_openvm_ir::{CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, LiftedInstr};
use rvr_openvm_lift::{ExtensionError, RvrExtension, RvrInstruction, RvrRuntimeExtension};

use crate::instruction::{decode_reg, Reg};

/// HINT_INPUT: make the next input record available through the hint stream.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HintInputInstr;

impl ExtInstr for HintInputInstr {
    fn opname(&self) -> &str {
        "hint_input"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        ctx.emit_call_without_page_flush("openvm_hint_input", &[]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(*self)
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// PRINT_STR: print a UTF-8 string from guest memory to host stdout.
#[derive(Debug, Clone)]
pub(crate) struct PrintStrInstr {
    pub(crate) ptr_reg: Reg,
    pub(crate) len_reg: Reg,
}

impl ExtInstr for PrintStrInstr {
    fn opname(&self) -> &str {
        "print_str"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let ptr = ctx.peek_var(self.ptr_reg);
        let len = ctx.peek_var(self.len_reg);
        ctx.emit_checked_call_without_page_flush("openvm_print_str", &[&ptr, &len]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// HINT_RANDOM: fill the hint stream with `num_words_reg * 8` random bytes
/// drawn from the host's persistent RNG.
#[derive(Debug, Clone)]
pub(crate) struct HintRandomInstr {
    pub(crate) num_words_reg: Reg,
}

impl ExtInstr for HintRandomInstr {
    fn opname(&self) -> &str {
        "hint_random"
    }

    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
        let num_words = ctx.peek_var(self.num_words_reg);
        ctx.emit_checked_call_without_page_flush("openvm_hint_random", &[&num_words]);
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }
}

/// RVR extension for RV64-specific phantom instructions.
pub struct Rv64PhantomExtension;

impl Rv64PhantomExtension {
    pub const fn new() -> Self {
        Self
    }
}

impl Default for Rv64PhantomExtension {
    fn default() -> Self {
        Self::new()
    }
}

impl RvrExtension for Rv64PhantomExtension {
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
        if insn.opcode.as_usize() != SystemOpcode::PHANTOM.global_opcode_usize() {
            return None;
        }
        let phantom = Rv64Phantom::from_repr((insn.c & 0xffff) as u16)?;
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
            instr,
            source_loc: None,
        }))
    }

    fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64_phantom_callbacks.h",
            include_str!("../c/rv64_phantom_callbacks.h"),
        )]
    }

    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![(
            "rv64_phantom_callbacks.c",
            include_str!("../c/rv64_phantom_callbacks.c"),
        )]
    }

    fn max_main_memory_pages_per_instruction(&self) -> usize {
        0
    }
}

/// Runtime hooks for RV64-specific phantom instructions.
pub struct Rv64PhantomRuntimeHooks;

impl RvrRuntimeExtension for Rv64PhantomRuntimeHooks {
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        let register_fn: RegisterRv64PhantomHostCallbacksFn = unsafe {
            let sym = lib
                .get::<RegisterRv64PhantomHostCallbacksFn>(b"register_rv64_phantom_host_callbacks")
                .map_err(|e| ExtensionError::HostCallbackRegistration(e.to_string()))?;
            *sym
        };
        let callbacks = Rv64PhantomHostCallbacks {
            hint_input: host_hint_input,
            print_str: host_print_str,
            hint_random: host_hint_random,
        };
        unsafe { register_fn(&callbacks) };
        Ok(())
    }
}

type RegisterRv64PhantomHostCallbacksFn = unsafe extern "C" fn(*const Rv64PhantomHostCallbacks);

/// Host callback table shared with `rv64_phantom_callbacks.c`.
#[repr(C)]
struct Rv64PhantomHostCallbacks {
    hint_input: extern "C" fn(*mut c_void),
    print_str: extern "C" fn(*mut c_void, u64, u64) -> bool,
    hint_random: extern "C" fn(*mut c_void, u64) -> bool,
}

/// Host callback for HINT_INPUT. Makes the next input record available without copying it.
pub(crate) extern "C" fn host_hint_input(ctx: *mut c_void) {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    if let Some(bytes) = io.input_stream.pop_front() {
        io.hint_stream.set_input(bytes);
    } else {
        io.hint_stream.clear();
    }
}

/// Host callback for PRINT_STR. Writes a guest-memory UTF-8 string to stdout.
extern "C" fn host_print_str(ctx: *mut c_void, ptr: u64, len: u64) -> bool {
    let io = unsafe { &*(ctx as *const OpenVmIoState<'_>) };
    let Some(range) = checked_mem_bounds_range(ptr, len) else {
        return false;
    };
    if len == 0 {
        return true;
    }
    if io.memory_ptr.is_null() {
        return false;
    }
    let slice = unsafe { std::slice::from_raw_parts(io.memory_ptr.add(range.start), range.len()) };
    if std::str::from_utf8(slice).is_err() {
        return false;
    }
    let _ = std::io::stdout().write_all(slice);
    let _ = std::io::stdout().flush();
    true
}

/// Host callback for HINT_RANDOM. Refills the hint stream with random RV64 words.
extern "C" fn host_hint_random(ctx: *mut c_void, num_words: u64) -> bool {
    let io = unsafe { &mut *(ctx as *mut OpenVmIoState<'_>) };
    let Ok(num_bytes) = hint_random_byte_len(num_words) else {
        return false;
    };
    io.hint_stream
        .try_set_hint_from_iter(
            num_bytes,
            repeat_with(|| io.rng.random::<u8>()).take(num_bytes),
        )
        .is_ok()
}

fn hint_random_byte_len(num_words: u64) -> Result<usize, &'static str> {
    let num_bytes = num_words
        .checked_mul(RV64_REGISTER_BYTES)
        .ok_or("byte count overflow")?;
    if num_bytes > MEM_SIZE as u64 {
        return Err("byte count exceeds resource limit");
    }
    Ok(num_bytes as usize)
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, ptr::null_mut};

    use openvm_circuit::arch::HintStream;
    use openvm_instructions::{instruction::Instruction, VmOpcode};
    use p3_baby_bear::BabyBear;
    use rand::{rngs::StdRng, SeedableRng};
    use test_case::test_case;

    use super::*;
    use crate::i::Rv64IExtension;

    fn phantom_instruction(phantom: Rv64Phantom) -> RvrInstruction {
        RvrInstruction::from_field(&Instruction::<BabyBear>::from_usize(
            VmOpcode::from_usize(SystemOpcode::PHANTOM.global_opcode_usize()),
            [8, 16, phantom as usize, 0, 0, 1, 0],
        ))
    }

    #[test]
    fn phantom_lifter_is_separate_from_rv64i() {
        let instruction = phantom_instruction(Rv64Phantom::PrintStr);

        assert!(Rv64IExtension.try_lift(&instruction, 0x100).is_none());
        let lifted = Rv64PhantomExtension
            .try_lift(&instruction, 0x100)
            .expect("RV64 phantom should lift");
        let LiftedInstr::Body(InstrAt { instr, .. }) = lifted else {
            panic!("expected body instruction");
        };
        assert_eq!(instr.opname(), "print_str");
    }

    #[test]
    fn print_str_rejects_invalid_utf8() {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = vec![0xff];
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

        assert!(!host_print_str(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            0,
            1,
        ));
    }

    #[test_case(u64::from(u32::MAX) + 1; "nonzero_upper_bits")]
    #[test_case(u64::MAX; "byte_count_overflow")]
    fn hint_random_rejects_invalid_full_rv64_count(num_words: u64) {
        let mut input_stream = VecDeque::new();
        let mut hint_stream = HintStream::default();
        hint_stream.set_hint(vec![0xa5]);
        let mut rng = StdRng::seed_from_u64(0);
        let mut memory = Vec::new();
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

        assert!(!host_hint_random(
            &mut io as *mut OpenVmIoState<'_> as *mut c_void,
            num_words,
        ));
        assert_eq!(io.hint_stream.remaining(), 1);
        let mut hint = [0; 1];
        io.hint_stream.copy_to_slice(&mut hint);
        assert_eq!(hint, [0xa5]);
    }
}
