use std::ffi::c_void;

use libloading::Library;
use openvm_instructions::exe::SparseMemoryImage;
use openvm_stark_backend::p3_field::PrimeField32;

use crate::arch::{
    interpreter::{AlignedBuf, PreComputeInstruction},
    ExecutionCtxTrait, SystemConfig,
};

mod metered_execute;
mod pure;

/// The assembly bridge build process requires the following tools:
/// GNU Binutils (provides `as` and `ar`)
/// Rust toolchain
/// Verify installation by `as --version`, `ar --version` and `cargo --version`
/// Refer to AOT.md for further clarification about AOT
///  
pub struct AotInstance<'a, F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    // SAFETY: this is not actually dead code, but `pre_compute_insns` contains raw pointer refers
    // to this buffer.
    #[allow(dead_code)]
    pre_compute_buf: AlignedBuf,
    lib: Library,
    pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]>,
    pc_start: u32,
}

type AsmRunFn = unsafe extern "C" fn(
    vm_exec_state_ptr: *mut c_void,
    pre_compute_insns_ptr: *const c_void,
    from_state_pc: u32,
    instret_left: u64,
);

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    fn push_external_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    push rbp\n";
        asm_str += "    push rbx\n";
        asm_str += "    push r12\n";
        asm_str += "    push r13\n";
        asm_str += "    push r15\n";
        // A dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    push r15\n";

        asm_str
    }

    fn pop_external_registers() -> String {
        let mut asm_str = String::new();
        // There was a dummy push to ensure the stack is 16 bytes aligned
        asm_str += "    pop r15\n";
        asm_str += "    pop r15\n";
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
        asm_str += "    push rax\n";
        asm_str += "    push rcx\n";
        asm_str += "    push rdx\n";
        asm_str += "    push r8\n";
        asm_str += "    push r9\n";
        asm_str += "    push r10\n";
        asm_str += "    push r11\n";
        // asm_str += &Self::push_xmm_regs();

        asm_str
    }

    fn pop_internal_registers() -> String {
        let mut asm_str = String::new();
        asm_str += "    pop r11\n";
        asm_str += "    pop r10\n";
        asm_str += "    pop r9\n";
        asm_str += "    pop r8\n";
        asm_str += "    pop rdx\n";
        asm_str += "    pop rcx\n";
        asm_str += "    pop rax\n";
        // asm_str += &Self::pop_xmm_regs();

        asm_str
    }

    /*
    fn sync_vm_registers() -> String  {
        let mut asm_str = String::new();
        for r in 0..16 {
            asm_str += &format!("");
            asm_str += &format!("   mov xmm{}, \n", r);
        }
    }
    */

    // r15 stores vm_register_address
    fn rv32_regs_to_xmm() -> String {
        let mut asm_str = String::new();

        for r in 0..16 {
            asm_str += &format!("   mov rdi, [r15 + 8*{r}]\n");
            asm_str += &format!("   pinsrq xmm{r}, rdi, 0\n");
        }

        asm_str
    }

    fn pop_address_space_start() -> String {
        let mut asm_str = String::new();
        // For byte alignment
        asm_str += "   pop rdi\n";
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

        for r in 0..16 {
            // at each iteration we save register 2r and 2r+1 of the guest mem to xmm
            asm_str += &format!("   movq [r15 + 8*{r}], xmm{r}\n");
        }

        asm_str
    }

    fn push_address_space_start() -> String {
        let mut asm_str = String::new();

        asm_str += "   pextrq rdi, xmm0, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm1, 1\n";
        asm_str += "   push rdi\n";
        asm_str += "   pextrq rdi, xmm2, 1\n";
        asm_str += "   push rdi\n";
        // For byte alignment
        asm_str += "   push rdi\n";

        asm_str
    }

    fn initialize_xmm_regs() -> String {
        let mut asm_str = String::new();
        asm_str += "    mov rax, 0\n";
        for r in 0..16 {
            asm_str += &format!("   pinsrq xmm{r}, rax, 0\n");
        }

        asm_str
    }

    pub fn to_i16(c: F) -> i16 {
        let c_u24 = (c.as_canonical_u64() & 0xFFFFFF) as u32;
        let c_i24 = ((c_u24 << 8) as i32) >> 8;
        c_i24 as i16
    }
}
