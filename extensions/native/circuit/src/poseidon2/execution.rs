use std::{
    borrow::{Borrow, BorrowMut},
    mem::size_of,
};

use openvm_circuit::{arch::*, system::memory::online::GuestMemory};
use openvm_circuit_primitives::AlignedBytesBorrow;
use openvm_instructions::{instruction::Instruction, program::DEFAULT_PC_STEP, LocalOpcode};
use openvm_native_compiler::{
    conversion::AS,
    Poseidon2Opcode::{COMP_POS2, PERM_POS2},
    VerifyBatchOpcode::VERIFY_BATCH,
};
use openvm_poseidon2_air::Poseidon2SubChip;
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::{ParallelIterator, ParallelSlice},
};

use super::chip::{compress, NativePoseidon2Executor};
use crate::poseidon2::CHUNK;

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct Pos2PreCompute<'a, F: Field, const SBOX_REGISTERS: usize> {
    subchip: &'a Poseidon2SubChip<F, SBOX_REGISTERS>,
    output_register: u32,
    input_register_1: u32,
    input_register_2: u32,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct VerifyBatchPreCompute<'a, F: Field, const SBOX_REGISTERS: usize> {
    subchip: &'a Poseidon2SubChip<F, SBOX_REGISTERS>,
    dim_register: u32,
    opened_register: u32,
    opened_length_register: u32,
    proof_id_ptr: u32,
    index_register: u32,
    commit_register: u32,
    opened_element_size: F,
}

impl<'a, F: PrimeField32, const SBOX_REGISTERS: usize> NativePoseidon2Executor<F, SBOX_REGISTERS> {
    #[inline(always)]
    fn pre_compute_pos2_impl(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        pos2_data: &mut Pos2PreCompute<'a, F, SBOX_REGISTERS>,
    ) -> Result<(), StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            ..
        } = inst;

        if opcode != PERM_POS2.global_opcode() && opcode != COMP_POS2.global_opcode() {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();

        if d != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }
        if e != AS::Native as u32 {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        *pos2_data = Pos2PreCompute {
            subchip: &self.subchip,
            output_register: a,
            input_register_1: b,
            input_register_2: c,
        };

        Ok(())
    }

    #[inline(always)]
    fn pre_compute_verify_batch_impl(
        &'a self,
        pc: u32,
        inst: &Instruction<F>,
        verify_batch_data: &mut VerifyBatchPreCompute<'a, F, SBOX_REGISTERS>,
    ) -> Result<(), StaticProgramError> {
        let &Instruction {
            opcode,
            a,
            b,
            c,
            d,
            e,
            f,
            g,
            ..
        } = inst;

        if opcode != VERIFY_BATCH.global_opcode() {
            return Err(StaticProgramError::InvalidInstruction(pc));
        }

        let a = a.as_canonical_u32();
        let b = b.as_canonical_u32();
        let c = c.as_canonical_u32();
        let d = d.as_canonical_u32();
        let e = e.as_canonical_u32();
        let f = f.as_canonical_u32();

        let opened_element_size_inv = g;
        // calc inverse fast assuming opened_element_size in {1, 4}
        let mut opened_element_size = F::ONE;
        while opened_element_size * opened_element_size_inv != F::ONE {
            opened_element_size += F::ONE;
        }

        *verify_batch_data = VerifyBatchPreCompute {
            subchip: &self.subchip,
            dim_register: a,
            opened_register: b,
            opened_length_register: c,
            proof_id_ptr: d,
            index_register: e,
            commit_register: f,
            opened_element_size,
        };

        Ok(())
    }
}

macro_rules! dispatch1 {
    (
        $execute_pos2_impl:ident,
        $execute_verify_batch_impl:ident,
        $executor:ident,
        $opcode:expr,
        $pc:ident,
        $inst:ident,
        $data:ident
    ) => {
        if $opcode == PERM_POS2.global_opcode() || $opcode == COMP_POS2.global_opcode() {
            let pos2_data: &mut Pos2PreCompute<F, SBOX_REGISTERS> = $data.borrow_mut();
            $executor.pre_compute_pos2_impl($pc, $inst, pos2_data)?;
            if $opcode == PERM_POS2.global_opcode() {
                Ok($execute_pos2_impl::<_, _, SBOX_REGISTERS, true>)
            } else {
                Ok($execute_pos2_impl::<_, _, SBOX_REGISTERS, false>)
            }
        } else {
            let verify_batch_data: &mut VerifyBatchPreCompute<F, SBOX_REGISTERS> =
                $data.borrow_mut();
            $executor.pre_compute_verify_batch_impl($pc, $inst, verify_batch_data)?;
            Ok($execute_verify_batch_impl::<_, _, SBOX_REGISTERS>)
        }
    };
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> Executor<F>
    for NativePoseidon2Executor<F, SBOX_REGISTERS>
{
    #[inline(always)]
    fn pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<Pos2PreCompute<F, SBOX_REGISTERS>>(),
            size_of::<VerifyBatchPreCompute<F, SBOX_REGISTERS>>(),
        )
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn pre_compute<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        dispatch1!(
            execute_pos2_e1_impl,
            execute_verify_batch_e1_impl,
            self,
            inst.opcode,
            pc,
            inst,
            data
        )
    }

    #[cfg(feature = "tco")]
    fn handler<Ctx: ExecutionCtxTrait>(
        &self,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        dispatch1!(
            execute_pos2_e1_handler,
            execute_verify_batch_e1_handler,
            self,
            inst.opcode,
            pc,
            inst,
            data
        )
    }
}

macro_rules! dispatch2 {
    (
        $execute_pos2_impl:ident,
        $execute_verify_batch_impl:ident,
        $executor:ident,
        $opcode:expr,
        $chip_idx:ident,
        $pc:ident,
        $inst:ident,
        $data:ident
    ) => {
        if $opcode == PERM_POS2.global_opcode() || $opcode == COMP_POS2.global_opcode() {
            let pre_compute: &mut E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>> =
                $data.borrow_mut();
            pre_compute.chip_idx = $chip_idx as u32;

            $executor.pre_compute_pos2_impl($pc, $inst, &mut pre_compute.data)?;
            if $opcode == PERM_POS2.global_opcode() {
                Ok($execute_pos2_impl::<_, _, SBOX_REGISTERS, true>)
            } else {
                Ok($execute_pos2_impl::<_, _, SBOX_REGISTERS, false>)
            }
        } else {
            let pre_compute: &mut E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>> =
                $data.borrow_mut();
            pre_compute.chip_idx = $chip_idx as u32;

            $executor.pre_compute_verify_batch_impl($pc, $inst, &mut pre_compute.data)?;
            Ok($execute_verify_batch_impl::<_, _, SBOX_REGISTERS>)
        }
    };
}

impl<F: PrimeField32, const SBOX_REGISTERS: usize> MeteredExecutor<F>
    for NativePoseidon2Executor<F, SBOX_REGISTERS>
{
    #[inline(always)]
    fn metered_pre_compute_size(&self) -> usize {
        std::cmp::max(
            size_of::<E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>>>(),
            size_of::<E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>>>(),
        )
    }

    #[cfg(not(feature = "tco"))]
    #[inline(always)]
    fn metered_pre_compute<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<ExecuteFunc<F, Ctx>, StaticProgramError> {
        dispatch2!(
            execute_pos2_e2_impl,
            execute_verify_batch_e2_impl,
            self,
            inst.opcode,
            chip_idx,
            pc,
            inst,
            data
        )
    }

    #[cfg(feature = "tco")]
    fn metered_handler<Ctx: MeteredExecutionCtxTrait>(
        &self,
        chip_idx: usize,
        pc: u32,
        inst: &Instruction<F>,
        data: &mut [u8],
    ) -> Result<Handler<F, Ctx>, StaticProgramError> {
        dispatch2!(
            execute_pos2_e2_handler,
            execute_verify_batch_e2_handler,
            self,
            inst.opcode,
            chip_idx,
            pc,
            inst,
            data
        )
    }
}

#[create_handler]
#[inline(always)]
unsafe fn execute_pos2_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &Pos2PreCompute<F, SBOX_REGISTERS> = pre_compute.borrow();
    execute_pos2_e12_impl::<_, _, SBOX_REGISTERS, IS_PERM>(pre_compute, instret, pc, exec_state);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_pos2_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _arg: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<Pos2PreCompute<F, SBOX_REGISTERS>> = pre_compute.borrow();
    let height = execute_pos2_e12_impl::<_, _, SBOX_REGISTERS, IS_PERM>(
        &pre_compute.data,
        instret,
        pc,
        exec_state,
    );
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[create_handler]
#[inline(always)]
unsafe fn execute_verify_batch_e1_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &VerifyBatchPreCompute<F, SBOX_REGISTERS> = pre_compute.borrow();
    // NOTE: using optimistic execution
    execute_verify_batch_e12_impl::<_, _, SBOX_REGISTERS, true>(
        pre_compute,
        instret,
        pc,
        exec_state,
    );
}

#[create_handler]
#[inline(always)]
unsafe fn execute_verify_batch_e2_impl<
    F: PrimeField32,
    CTX: MeteredExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
>(
    pre_compute: &[u8],
    instret: &mut u64,
    pc: &mut u32,
    _instret_end: u64,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) {
    let pre_compute: &E2PreCompute<VerifyBatchPreCompute<F, SBOX_REGISTERS>> = pre_compute.borrow();
    // NOTE: using optimistic execution
    let height = execute_verify_batch_e12_impl::<_, _, SBOX_REGISTERS, true>(
        &pre_compute.data,
        instret,
        pc,
        exec_state,
    );
    exec_state
        .ctx
        .on_height_change(pre_compute.chip_idx as usize, height);
}

#[inline(always)]
unsafe fn execute_pos2_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
    const IS_PERM: bool,
>(
    pre_compute: &Pos2PreCompute<F, SBOX_REGISTERS>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let subchip = pre_compute.subchip;

    let [output_pointer]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.output_register);
    let [input_pointer_1]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.input_register_1);
    let [input_pointer_2] = if IS_PERM {
        [input_pointer_1 + F::from_canonical_usize(CHUNK)]
    } else {
        exec_state.vm_read(AS::Native as u32, pre_compute.input_register_2)
    };

    let data_1: [F; CHUNK] =
        exec_state.vm_read(AS::Native as u32, input_pointer_1.as_canonical_u32());
    let data_2: [F; CHUNK] =
        exec_state.vm_read(AS::Native as u32, input_pointer_2.as_canonical_u32());

    let p2_input = std::array::from_fn(|i| {
        if i < CHUNK {
            data_1[i]
        } else {
            data_2[i - CHUNK]
        }
    });
    let output = subchip.permute(p2_input);
    let output_pointer_u32 = output_pointer.as_canonical_u32();

    exec_state.vm_write::<F, CHUNK>(
        AS::Native as u32,
        output_pointer_u32,
        &std::array::from_fn(|i| output[i]),
    );
    if IS_PERM {
        exec_state.vm_write::<F, CHUNK>(
            AS::Native as u32,
            output_pointer_u32 + CHUNK as u32,
            &std::array::from_fn(|i| output[i + CHUNK]),
        );
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    1
}

#[inline(always)]
unsafe fn execute_verify_batch_e12_impl<
    F: PrimeField32,
    CTX: ExecutionCtxTrait,
    const SBOX_REGISTERS: usize,
    const OPTIMISTIC: bool,
>(
    pre_compute: &VerifyBatchPreCompute<F, SBOX_REGISTERS>,
    instret: &mut u64,
    pc: &mut u32,
    exec_state: &mut VmExecState<F, GuestMemory, CTX>,
) -> u32 {
    let subchip = pre_compute.subchip;
    let opened_element_size = pre_compute.opened_element_size;

    let [proof_id]: [F; 1] = exec_state.host_read(AS::Native as u32, pre_compute.proof_id_ptr);
    let [dim_base_pointer]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.dim_register);
    let dim_base_pointer_u32 = dim_base_pointer.as_canonical_u32();
    let [opened_base_pointer]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.opened_register);
    let opened_base_pointer_u32 = opened_base_pointer.as_canonical_u32();
    let [opened_length]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.opened_length_register);
    let [index_base_pointer]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.index_register);
    let index_base_pointer_u32 = index_base_pointer.as_canonical_u32();
    let [commit_pointer]: [F; 1] =
        exec_state.vm_read(AS::Native as u32, pre_compute.commit_register);
    let commit: [F; CHUNK] =
        exec_state.vm_read(AS::Native as u32, commit_pointer.as_canonical_u32());

    let opened_length = opened_length.as_canonical_u32() as usize;

    let initial_log_height = {
        let [height]: [F; 1] = exec_state.host_read(AS::Native as u32, dim_base_pointer_u32);
        height.as_canonical_u32()
    };

    let mut log_height = initial_log_height as i32;
    let mut sibling_index = 0;
    let mut opened_index = 0;
    let mut height = 0;

    let mut root = [F::ZERO; CHUNK];
    let sibling_proof: Vec<[F; CHUNK]> = {
        let proof_idx = proof_id.as_canonical_u32() as usize;
        exec_state.streams.hint_space[proof_idx]
            .par_chunks(CHUNK)
            .map(|c| c.try_into().unwrap())
            .collect()
    };

    while log_height >= 0 {
        if opened_index < opened_length
            && exec_state.host_read::<F, 1>(
                AS::Native as u32,
                dim_base_pointer_u32 + opened_index as u32,
            )[0] == F::from_canonical_u32(log_height as u32)
        {
            let initial_opened_index = opened_index;

            let mut row_pointer = 0;
            let mut row_end = 0;

            let mut rolling_hash = [F::ZERO; 2 * CHUNK];

            let mut is_first_in_segment = true;

            loop {
                let mut cells_len = 0;
                for chunk_elem in rolling_hash.iter_mut().take(CHUNK) {
                    if is_first_in_segment || row_pointer == row_end {
                        if is_first_in_segment {
                            is_first_in_segment = false;
                        } else {
                            opened_index += 1;
                            if opened_index == opened_length
                                || exec_state.host_read::<F, 1>(
                                    AS::Native as u32,
                                    dim_base_pointer_u32 + opened_index as u32,
                                )[0] != F::from_canonical_u32(log_height as u32)
                            {
                                break;
                            }
                        }
                        let [new_row_pointer, row_len]: [F; 2] = exec_state.vm_read(
                            AS::Native as u32,
                            opened_base_pointer_u32 + 2 * opened_index as u32,
                        );
                        row_pointer = new_row_pointer.as_canonical_u32() as usize;
                        row_end = row_pointer
                            + (opened_element_size * row_len).as_canonical_u32() as usize;
                    }
                    let [value]: [F; 1] = exec_state.vm_read(AS::Native as u32, row_pointer as u32);
                    cells_len += 1;
                    *chunk_elem = value;
                    row_pointer += 1;
                }
                if cells_len == 0 {
                    break;
                }
                height += 1;
                if !OPTIMISTIC {
                    subchip.permute_mut(&mut rolling_hash);
                }
                if cells_len < CHUNK {
                    break;
                }
            }

            let final_opened_index = opened_index - 1;
            let [height_check]: [F; 1] = exec_state.host_read(
                AS::Native as u32,
                dim_base_pointer_u32 + initial_opened_index as u32,
            );
            assert_eq!(height_check, F::from_canonical_u32(log_height as u32));
            let [height_check]: [F; 1] = exec_state.host_read(
                AS::Native as u32,
                dim_base_pointer_u32 + final_opened_index as u32,
            );
            assert_eq!(height_check, F::from_canonical_u32(log_height as u32));

            if !OPTIMISTIC {
                let hash: [F; CHUNK] = std::array::from_fn(|i| rolling_hash[i]);

                let new_root = if log_height as u32 == initial_log_height {
                    hash
                } else {
                    let (_, new_root) = compress(subchip, root, hash);
                    new_root
                };
                root = new_root;
            }
            height += 1;
        }

        if log_height != 0 {
            let [sibling_is_on_right]: [F; 1] = exec_state.vm_read(
                AS::Native as u32,
                index_base_pointer_u32 + sibling_index as u32,
            );
            let sibling_is_on_right = sibling_is_on_right == F::ONE;
            let sibling = sibling_proof[sibling_index];
            if !OPTIMISTIC {
                let (_, new_root) = if sibling_is_on_right {
                    compress(subchip, sibling, root)
                } else {
                    compress(subchip, root, sibling)
                };
                root = new_root;
            }
            height += 1;
        }

        log_height -= 1;
        sibling_index += 1;
    }

    if !OPTIMISTIC {
        assert_eq!(commit, root);
    }

    *pc = pc.wrapping_add(DEFAULT_PC_STEP);
    *instret += 1;

    height
}
