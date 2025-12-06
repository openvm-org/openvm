
use std::{
    borrow::{Borrow},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    program::DEFAULT_PC_STEP,
    riscv::{RV32_MEMORY_AS, RV32_REGISTER_AS},
};
use openvm_stark_backend::p3_field::PrimeField32;
use p3_keccak_air::NUM_ROUNDS;

use super::KECCAK_WORD_SIZE;

use tiny_keccak::{Hasher, Keccak};

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct KeccakPreCompute {
    a: u8,
    b: u8,
    c: u8,
}

#[inline(always)]
unsafe fn execute_e12_impl<F: PrimeField32, CTX: ExecutionCtxTrait, const IS_E1: bool>(
    pre_compute: &KeccakPreCompute,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let dst = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.a as u32);
    let src = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.b as u32);
    let len = exec_state.vm_read(RV32_REGISTER_AS, pre_compute.c as u32);
    let dst_u32 = u32::from_le_bytes(dst);
    let src_u32 = u32::from_le_bytes(src);
    let len_u32 = u32::from_le_bytes(len);

    let (output, height) = if IS_E1 {
        // SAFETY: RV32_MEMORY_AS is memory address space of type u8
        let message = exec_state.vm_read_slice(RV32_MEMORY_AS, src_u32, len_u32 as usize);
        let mut hasher = Keccak::v256();
        hasher.update(message);
        

        let output = keccak256(message);
        (output, 0)
    } else {
        let num_reads = (len_u32 as usize).div_ceil(KECCAK_WORD_SIZE);
        let message: Vec<_> = (0..num_reads)
            .flat_map(|i| {
                exec_state.vm_read::<u8, KECCAK_WORD_SIZE>(
                    RV32_MEMORY_AS,
                    src_u32 + (i * KECCAK_WORD_SIZE) as u32,
                )
            })
            .collect();
        let output = keccak256(&message[..len_u32 as usize]);
        let height = (num_keccak_f(len_u32 as usize) * NUM_ROUNDS) as u32;
        (output, height)
    };
    exec_state.vm_write(RV32_MEMORY_AS, dst_u32, &output);

    let pc = exec_state.pc();
    exec_state.set_pc(pc.wrapping_add(DEFAULT_PC_STEP));

    height
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e1_impl<F: PrimeField32, CTX: ExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &KeccakPreCompute =
        std::slice::from_raw_parts(pre_compute, size_of::<KeccakPreCompute>()).borrow();
    execute_e12_impl::<F, CTX, true>(pre_compute, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_e2_impl<F: PrimeField32, CTX: MeteredExecutionCtxTrait>(
    pre_compute: *const u8,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<KeccakPreCompute> =
        std::slice::from_raw_parts(pre_compute, size_of::<E2PreCompute<KeccakPreCompute>>())
            .borrow();
    let height = execute_e12_impl::<F, CTX, false>(&pre_compute.data, exec_state);
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}
