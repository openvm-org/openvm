use std::{ffi::c_void, io::Write, process::Command};

use libloading::Library;
use openvm_instructions::{exe::SparseMemoryImage, program::DEFAULT_PC_STEP};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        aot::common::{sync_gpr_to_xmm, sync_xmm_to_gpr, REG_AS2_PTR},
        execution_mode::ExecutionCtx,
        interpreter::{AlignedBuf, PreComputeInstruction},
        ExecutionCtxTrait, StaticProgramError, Streams, SystemConfig, VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

pub mod common;
mod metered_execute;
mod pure;

/// The assembly bridge build process requires the following tools:
/// GNU Binutils (provides `as` and `ar`)
/// Rust toolchain
/// Verify installation by `as --version`, `ar --version` and `cargo --version`
/// Refer to AOT.md for further clarification about AOT
///  
pub struct AotInstance<F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    // SAFETY: this is not actually dead code, but `pre_compute_insns` contains raw pointer refers
    // to this buffer.
    #[allow(dead_code)]
    pre_compute_buf: AlignedBuf,
    lib: Library,
    pre_compute_insns: Vec<PreComputeInstruction<F, Ctx>>,
    pc_start: u32,
}

type AsmRunFn = unsafe extern "C" fn(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    from_state_pc: u32,
    instret_left: u64,
);

impl<F, Ctx> AotInstance<F, Ctx>
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams<F>>) -> VmState<F> {
        VmState::initial(
            &self.system_config,
            &self.init_memory,
            self.pc_start,
            inputs,
        )
    }

    fn push_external_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    push rbp\n";
        asm_str += "    push rbx\n";
        asm_str += "    push r12\n";
        asm_str += "    push r13\n";
        asm_str += "    push r14\n";
        // A dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    push r15\n";

        asm_str
    }

    fn pop_external_registers() -> String {
        let mut asm_str = String::new();
        // There was a dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    pop r15\n";
        asm_str += "    pop r14\n";
        asm_str += "    pop r13\n";
        asm_str += "    pop r12\n";
        asm_str += "    pop rbx\n";
        asm_str += "    pop rbp\n";

        asm_str
    }

    #[allow(dead_code)]
    fn debug_cur_string(str: &String) {
        println!("DEBUG");
        println!("{str}");
    }

    #[allow(dead_code)]
    fn push_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    sub rsp, 16*16";
        asm_str += "    movaps [rsp + 0*16], xmm0\n";
        asm_str += "    movaps [rsp + 1*16], xmm1\n";
        asm_str += "    movaps [rsp + 2*16], xmm2\n";
        asm_str += "    movaps [rsp + 3*16], xmm3\n";
        asm_str += "    movaps [rsp + 4*16], xmm4\n";
        asm_str += "    movaps [rsp + 5*16], xmm5\n";
        asm_str += "    movaps [rsp + 6*16], xmm6\n";
        asm_str += "    movaps [rsp + 7*16], xmm7\n";
        asm_str += "    movaps [rsp + 8*16], xmm8\n";
        asm_str += "    movaps [rsp + 9*16], xmm9\n";
        asm_str += "    movaps [rsp + 10*16], xmm10\n";
        asm_str += "    movaps [rsp + 11*16], xmm11\n";
        asm_str += "    movaps [rsp + 12*16], xmm12\n";
        asm_str += "    movaps [rsp + 13*16], xmm13\n";
        asm_str += "    movaps [rsp + 14*16], xmm14\n";
        asm_str += "    movaps [rsp + 15*16], xmm15\n";

        asm_str
    }

    #[allow(dead_code)]
    fn pop_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    movaps xmm0, [rsp + 0*16]\n";
        asm_str += "    movaps xmm1, [rsp + 1*16]\n";
        asm_str += "    movaps xmm2, [rsp + 2*16]\n";
        asm_str += "    movaps xmm3, [rsp + 3*16]\n";
        asm_str += "    movaps xmm4, [rsp + 4*16]\n";
        asm_str += "    movaps xmm5, [rsp + 5*16]\n";
        asm_str += "    movaps xmm6, [rsp + 6*16]\n";
        asm_str += "    movaps xmm7, [rsp + 7*16]\n";
        asm_str += "    movaps xmm8, [rsp + 8*16]\n";
        asm_str += "    movaps xmm9, [rsp + 9*16]\n";
        asm_str += "    movaps xmm10, [rsp + 10*16]\n";
        asm_str += "    movaps xmm11, [rsp + 11*16]\n";
        asm_str += "    movaps xmm12, [rsp + 12*16]\n";
        asm_str += "    movaps xmm13, [rsp + 13*16]\n";
        asm_str += "    movaps xmm14, [rsp + 14*16]\n";
        asm_str += "    movaps xmm15, [rsp + 15*16]\n";
        asm_str += "    add rsp, 16*16\n";

        asm_str
    }

    fn push_internal_registers() -> String {
        let mut asm_str = String::new();

        asm_str += "    push rcx\n";
        asm_str += "    push rdx\n";
        asm_str += "    push rsi\n";
        asm_str += "    push rdi\n";
        asm_str += "    push r8\n";
        asm_str += "    push r9\n";
        asm_str += "    push r10\n";
        asm_str += "    push r11\n";
        asm_str += "    push rax\n";

        asm_str
    }

    fn pop_internal_registers() -> String {
        let mut asm_str = String::new();

        asm_str += "    pop rax\n";
        asm_str += "    pop r11\n";
        asm_str += "    pop r10\n";
        asm_str += "    pop r9\n";
        asm_str += "    pop r8\n";
        asm_str += "    pop rdi\n";
        asm_str += "    pop rsi\n";
        asm_str += "    pop rdx\n";
        asm_str += "    pop rcx\n";

        asm_str
    }

    // r15 stores vm_register_address
    fn rv32_regs_to_xmm() -> String {
        let mut asm_str = String::new();

        asm_str += &format!("    push {REG_AS2_PTR}\n");
        asm_str += &format!("    pextrq {REG_AS2_PTR}, xmm0, 1\n");

        for r in 0..16 {
            asm_str += &format!("   mov rdi, [{REG_AS2_PTR} + 8*{r}]\n");
            asm_str += &format!("   pinsrq xmm{r}, rdi, 0\n");
        }

        asm_str += &format!("    pop {REG_AS2_PTR}\n");

        asm_str += &sync_xmm_to_gpr();

        asm_str
    }

    fn pop_address_space_start() -> String {
        let mut asm_str = String::new();
        // SAFETY: pay attention to byte alignment.
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm3, rdi, 1\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm2, rdi, 1\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm1, rdi, 1\n";
        asm_str += "   pop rdi\n";
        asm_str += "   pinsrq xmm0, rdi, 1\n";
        asm_str
    }

    fn xmm_to_rv32_regs() -> String {
        let mut asm_str = String::new();

        asm_str += &sync_gpr_to_xmm();

        asm_str += &format!("    push {REG_AS2_PTR}\n");
        asm_str += &format!("    pextrq {REG_AS2_PTR}, xmm0, 1\n");

        for r in 0..16 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm_str += &format!("   movq [{REG_AS2_PTR} + 8*{r}], xmm{r}\n");
        }

        asm_str += &format!("    pop {REG_AS2_PTR}\n");

        asm_str
    }

    fn push_address_space_start() -> String {
        let mut asm_str = String::new();

        // SAFETY: pay attention to byte alignment.
        asm_str += "   pextrq rdi, xmm0, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm1, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm2, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm3, 1\n";
        asm_str += "   push rdi\n";

        asm_str
    }

    pub fn to_i16(c: F) -> i16 {
        let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
        let c_i24 = ((c_u24 << 8) as i32) >> 8;
        c_i24 as i16
    }
}

pub(crate) fn asm_to_lib(asm_source: &str) -> Result<Library, StaticProgramError> {
    let start = std::time::Instant::now();
    // Create a temporary file for the .s file.
    let src_file = tempfile::Builder::new()
        .prefix("asm_x86_run")
        .suffix(".s")
        .tempfile()
        .expect("Failed to create temporary file for asm_x86_run .s file");
    src_file
        .as_file()
        .write(asm_source.as_bytes())
        .map_err(|e| StaticProgramError::FailToWriteTemporaryFile { err: e.to_string() })?;
    let src_path = src_file.into_temp_path();

    // Create a temporary file for the .so file.
    let lib_path = tempfile::Builder::new()
        .prefix("asm_x86_run")
        .suffix(".so")
        .tempfile()
        .map_err(|e| StaticProgramError::FailToCreateTemporaryFile { err: e.to_string() })?
        .into_temp_path();

    // gcc -fPIC -Wl,-z,noexecstack -shared asm_x86_run.s -o asm_x86_run.so
    let status = Command::new("gcc")
        .arg("-fPIC")
        .arg("-Wl,-z,noexecstack")
        .arg("-shared")
        .arg(&src_path)
        .arg("-o")
        .arg(&lib_path)
        .status()
        .map_err(|e| StaticProgramError::FailToGenerateDynamicLibrary { err: e.to_string() })?;
    if !status.success() {
        return Err(StaticProgramError::FailToGenerateDynamicLibrary {
            err: status.to_string(),
        });
    }

    let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };
    tracing::trace!(
        "Time taken to build and load .so for AotInstance metered execution: {}ms",
        start.elapsed().as_millis()
    );
    Ok(lib)
}

unsafe extern "C" fn should_suspend_shim<F, Ctx: ExecutionCtxTrait>(
    state_ptr: *mut c_void,
) -> bool {
    let state = &mut *(state_ptr as *mut VmExecState<F, GuestMemory, Ctx>);
    VmExecState::<F, GuestMemory, Ctx>::should_suspend(state)
}

unsafe extern "C" fn set_pc_shim<F, Ctx: ExecutionCtxTrait>(state_ptr: *mut c_void, pc: u32) {
    let state = &mut *(state_ptr as *mut VmExecState<F, GuestMemory, Ctx>);
    state.vm_state.set_pc(pc);
}

// only needed for pure execution
unsafe extern "C" fn set_instret_left_shim<F>(state_ptr: *mut c_void, instret_left: u64) {
    let state = &mut *(state_ptr as *mut VmExecState<F, GuestMemory, ExecutionCtx>);
    state.ctx.instret_left = instret_left;
}

pub(crate) extern "C" fn extern_handler<F, Ctx: ExecutionCtxTrait, const E1: bool>(
    state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    cur_pc: u32,
) -> u32 {
    let vm_exec_state_ref = unsafe { &mut *(state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    vm_exec_state_ref.set_pc(cur_pc);

    // pointer to the first element of `pre_compute_insns`
    let pre_compute_insns_base_ptr = pre_compute_insns_ptr as *const PreComputeInstruction<F, Ctx>;
    let pc_idx = (cur_pc / DEFAULT_PC_STEP) as usize;
    let pre_compute_insns = unsafe { &*pre_compute_insns_base_ptr.add(pc_idx) };
    unsafe {
        (pre_compute_insns.handler)(pre_compute_insns.pre_compute, vm_exec_state_ref);
    };

    match vm_exec_state_ref.exit_code {
        Ok(None) => 0,
        _ => 1,
    }
}

extern "C" fn get_vm_address_space_addr<F, Ctx: ExecutionCtxTrait>(
    exec_state_ptr: *mut c_void,
    addr_space: u64,
) -> *mut u64 {
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    let ptr = &vm_exec_state_ref.vm_state.memory.memory.mem[addr_space as usize];
    ptr.as_ptr() as *mut u64 // mut u64 because we want to write 8 bytes at a time
}

extern "C" fn get_vm_pc_ptr<F, Ctx: ExecutionCtxTrait>(exec_state_ptr: *mut c_void) -> *mut u64 {
    let vm_exec_state_ref =
        unsafe { &mut *(exec_state_ptr as *mut VmExecState<F, GuestMemory, Ctx>) };
    // since pc is the first element of the vm_state field and we use `repr(C)`
    // hence `ptr` will be equal to the address of pc in vm_state
    let state = &mut vm_exec_state_ref.vm_state;
    let ptr = state.pc_mut() as *mut u32;

    ptr as *mut u64
}
