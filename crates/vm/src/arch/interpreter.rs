use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    borrow::{Borrow, BorrowMut},
    ptr::NonNull,
};

use itertools::Itertools;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    exe::VmExe,
    instruction::Instruction,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SystemOpcode,
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info_span;

use crate::{
    arch::{
        create_memory_image,
        execution_mode::{E1ExecutionCtx, E2ExecutionCtx},
        ExecuteFunc, ExecutionError, ExecutorInventory, ExecutorInventoryError, ExitCode,
        InsExecutorE1, InsExecutorE2, PreComputeInstruction, Streams, SystemConfig,
        VmExecutionConfig, VmSegmentState,
    },
    system::memory::online::GuestMemory,
};

/// VM pure executor(E1/E2 executor) which doesn't consider trace generation.
/// Note: This executor doesn't hold any VM state and can be used for multiple execution.
pub struct InterpretedInstance<F, E> {
    system_config: SystemConfig,
    // TODO: don't need exe, just precompute handlers
    exe: VmExe<F>,
    // TODO: don't clone config or inventory, hold reference
    inventory: ExecutorInventory<E>,
    e1_pre_compute_max_size: usize,
    e2_pre_compute_max_size: usize,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct TerminatePreCompute {
    exit_code: u32,
}

macro_rules! execute_with_metrics {
    ($span:literal, $program:expr, $vm_state:expr, $pre_compute_insts:expr) => {{
        #[cfg(feature = "metrics")]
        let start = std::time::Instant::now();
        #[cfg(feature = "metrics")]
        let start_instret = $vm_state.instret;

        info_span!($span).in_scope(|| unsafe {
            execute_impl($program, $vm_state, $pre_compute_insts);
        });

        #[cfg(feature = "metrics")]
        {
            let elapsed = start.elapsed();
            let insns = $vm_state.instret - start_instret;
            metrics::counter!("insns").absolute(insns);
            metrics::gauge!(concat!($span, "_insn_mi/s"))
                .set(insns as f64 / elapsed.as_micros() as f64);
        }
    }};
}

impl<F, E> InterpretedInstance<F, E>
where
    F: PrimeField32,
    E: InsExecutorE1<F> + InsExecutorE2<F>,
{
    pub fn new<VC>(vm_config: VC, exe: impl Into<VmExe<F>>) -> Result<Self, ExecutorInventoryError>
    where
        VC: VmExecutionConfig<F, Executor = E> + AsRef<SystemConfig>,
    {
        let exe = exe.into();
        let program = &exe.program;
        let inventory = vm_config.create_executors()?;
        let e1_pre_compute_max_size = get_pre_compute_max_size(program, &inventory);
        let e2_pre_compute_max_size = get_e2_pre_compute_max_size(program, &inventory);
        Ok(Self {
            exe,
            system_config: vm_config.as_ref().clone(),
            inventory,
            e1_pre_compute_max_size,
            e2_pre_compute_max_size,
        })
    }

    /// Execute the VM program with the given execution control and inputs. Returns the final VM
    /// state with the `ExecutionControl` context.
    pub fn execute<Ctx: E1ExecutionCtx>(
        &self,
        ctx: Ctx,
        inputs: impl Into<Streams<F>>,
    ) -> Result<VmSegmentState<F, GuestMemory, Ctx>, ExecutionError> {
        // Initialize the chip complex
        let mut vm_state = self.init_vm_state(ctx, inputs);

        // Start execution
        let program = &self.exe.program;
        let pre_compute_max_size = self.e1_pre_compute_max_size;
        let mut pre_compute_buf = self.alloc_pre_compute_buf(pre_compute_max_size);
        let mut split_pre_compute_buf =
            self.split_pre_compute_buf(&mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insts = get_pre_compute_instructions::<_, _, Ctx>(
            program,
            &self.inventory,
            &mut split_pre_compute_buf,
        )?;
        execute_with_metrics!("execute_e1", program, &mut vm_state, &pre_compute_insts);
        if vm_state.exit_code.is_err() {
            Err(vm_state.exit_code.err().unwrap())
        } else {
            check_exit_code(&vm_state)?;
            Ok(vm_state)
        }
    }

    /// Execute the VM program with the given execution control and inputs. Returns the final VM
    /// state with the `ExecutionControl` context.
    pub fn execute_e2<Ctx: E2ExecutionCtx>(
        &self,
        ctx: Ctx,
        inputs: impl Into<Streams<F>>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<VmSegmentState<F, GuestMemory, Ctx>, ExecutionError> {
        // Initialize the chip complex
        let mut vm_state = self.init_vm_state(ctx, inputs);

        // Start execution
        let program = &self.exe.program;
        let pre_compute_max_size = self.e2_pre_compute_max_size;
        let mut pre_compute_buf = self.alloc_pre_compute_buf(pre_compute_max_size);
        let mut split_pre_compute_buf =
            self.split_pre_compute_buf(&mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insts = get_e2_pre_compute_instructions::<_, _, Ctx>(
            program,
            &self.inventory,
            executor_idx_to_air_idx,
            &mut split_pre_compute_buf,
        )?;
        execute_with_metrics!(
            "execute_metered",
            program,
            &mut vm_state,
            &pre_compute_insts
        );
        if vm_state.exit_code.is_err() {
            Err(vm_state.exit_code.err().unwrap())
        } else {
            check_exit_code(&vm_state)?;
            Ok(vm_state)
        }
    }

    pub fn init_vm_state<Ctx: E1ExecutionCtx>(
        &self,
        ctx: Ctx,
        inputs: impl Into<Streams<F>>,
    ) -> VmSegmentState<F, GuestMemory, Ctx> {
        let memory_config = &self.system_config.memory_config;
        let memory = create_memory_image(memory_config, self.exe.init_memory.clone());

        VmSegmentState::new(
            0,
            self.exe.pc_start,
            memory,
            inputs.into(),
            StdRng::seed_from_u64(0),
            ctx,
        )
    }

    #[inline(always)]
    fn alloc_pre_compute_buf(&self, pre_compute_max_size: usize) -> AlignedBuf {
        let program = &self.exe.program;
        let program_len = program.instructions_and_debug_infos.len();
        let buf_len = program_len * pre_compute_max_size;
        AlignedBuf::uninit(buf_len, pre_compute_max_size)
    }

    #[inline(always)]
    fn split_pre_compute_buf<'a>(
        &self,
        pre_compute_buf: &'a mut AlignedBuf,
        pre_compute_max_size: usize,
    ) -> Vec<&'a mut [u8]> {
        let program = &self.exe.program;
        let program_len = program.instructions_and_debug_infos.len();
        let buf_len = program_len * pre_compute_max_size;
        let mut pre_compute_buf_ptr =
            unsafe { std::slice::from_raw_parts_mut(pre_compute_buf.ptr, buf_len) };
        let mut split_pre_compute_buf = Vec::with_capacity(program_len);
        for _ in 0..program_len {
            let (first, last) = pre_compute_buf_ptr.split_at_mut(pre_compute_max_size);
            pre_compute_buf_ptr = last;
            split_pre_compute_buf.push(first);
        }
        split_pre_compute_buf
    }
}

#[inline(never)]
unsafe fn execute_impl<F: PrimeField32, Ctx: E1ExecutionCtx>(
    program: &Program<F>,
    vm_state: &mut VmSegmentState<F, GuestMemory, Ctx>,
    fn_ptrs: &[PreComputeInstruction<F, Ctx>],
) {
    // let start = std::time::Instant::now();
    while vm_state
        .exit_code
        .as_ref()
        .is_ok_and(|exit_code| exit_code.is_none())
    {
        if Ctx::should_suspend(vm_state) {
            break;
        }
        let pc_index = get_pc_index(program, vm_state.pc).unwrap();
        let inst = &fn_ptrs[pc_index];
        unsafe { (inst.handler)(inst.pre_compute, vm_state) };
    }
    if vm_state
        .exit_code
        .as_ref()
        .is_ok_and(|exit_code| exit_code.is_some())
    {
        Ctx::on_terminate(vm_state);
    }
    // println!("execute time: {}ms", start.elapsed().as_millis());
}

fn get_pc_index<F: Field>(program: &Program<F>, pc: u32) -> Result<usize, ExecutionError> {
    let pc_base = program.pc_base;
    let pc_index = ((pc - pc_base) / DEFAULT_PC_STEP) as usize;
    if !(0..program.len()).contains(&pc_index) {
        return Err(ExecutionError::PcOutOfBounds {
            pc,
            pc_base,
            program_len: program.len(),
        });
    }
    Ok(pc_index)
}

/// Bytes allocated according to the given Layout
pub struct AlignedBuf {
    pub ptr: *mut u8,
    pub layout: Layout,
}

impl AlignedBuf {
    /// Allocate a new buffer whose start address is aligned to `align` bytes.
    /// *NOTE* if `len` is zero then a creates new `NonNull` that is dangling and 16-byte aligned.
    pub fn uninit(len: usize, align: usize) -> Self {
        let layout = Layout::from_size_align(len, align).unwrap();
        if layout.size() == 0 {
            return Self {
                ptr: NonNull::<u128>::dangling().as_ptr() as *mut u8,
                layout,
            };
        }
        // SAFETY: `len` is nonzero
        let ptr = unsafe { alloc(layout) };
        if ptr.is_null() {
            handle_alloc_error(layout);
        }
        AlignedBuf { ptr, layout }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.layout.size() != 0 {
            unsafe {
                dealloc(self.ptr, self.layout);
            }
        }
    }
}

unsafe fn terminate_execute_e12_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, GuestMemory, CTX>,
) {
    let pre_compute: &TerminatePreCompute = pre_compute.borrow();
    vm_state.instret += 1;
    vm_state.exit_code = Ok(Some(pre_compute.exit_code));
}

fn get_pre_compute_max_size<F, E: InsExecutorE1<F>>(
    program: &Program<F>,
    inventory: &ExecutorInventory<E>,
) -> usize {
    program
        .instructions_and_debug_infos
        .iter()
        .map(|inst_opt| {
            if let Some((inst, _)) = inst_opt {
                if let Some(size) = system_opcode_pre_compute_size(inst) {
                    size
                } else {
                    inventory
                        .get_executor(inst.opcode)
                        .map(|executor| executor.pre_compute_size())
                        .unwrap()
                }
            } else {
                0
            }
        })
        .max()
        .unwrap()
        .next_power_of_two()
}

fn get_e2_pre_compute_max_size<F: PrimeField32, E: InsExecutorE2<F>>(
    program: &Program<F>,
    inventory: &ExecutorInventory<E>,
) -> usize {
    program
        .instructions_and_debug_infos
        .iter()
        .map(|inst_opt| {
            if let Some((inst, _)) = inst_opt {
                if let Some(size) = system_opcode_pre_compute_size(inst) {
                    size
                } else {
                    inventory
                        .get_executor(inst.opcode)
                        .map(|executor| executor.e2_pre_compute_size())
                        .unwrap()
                }
            } else {
                0
            }
        })
        .max()
        .unwrap()
        .next_power_of_two()
}

fn system_opcode_pre_compute_size<F>(inst: &Instruction<F>) -> Option<usize> {
    if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
        return Some(size_of::<TerminatePreCompute>());
    }
    None
}

fn get_pre_compute_instructions<'a, F: PrimeField32, E: InsExecutorE1<F>, Ctx: E1ExecutionCtx>(
    program: &'a Program<F>,
    inventory: &'a ExecutorInventory<E>,
    pre_compute: &'a mut [&mut [u8]],
) -> Result<Vec<PreComputeInstruction<'a, F, Ctx>>, ExecutionError> {
    program
        .instructions_and_debug_infos
        .iter()
        .zip_eq(pre_compute.iter_mut())
        .enumerate()
        .map(|(i, (inst_opt, buf))| {
            let buf: &mut [u8] = buf;
            let pre_inst = if let Some((inst, _)) = inst_opt {
                tracing::trace!("get_e2_pre_compute_instruction {inst:?}");
                let pc = program.pc_base + i as u32 * DEFAULT_PC_STEP;
                if let Some(handler) = get_system_opcode_handler(inst, buf) {
                    PreComputeInstruction {
                        handler,
                        pre_compute: buf,
                    }
                } else if let Some(executor) = inventory.get_executor(inst.opcode) {
                    PreComputeInstruction {
                        handler: executor.pre_compute_e1(pc, inst, buf)?,
                        pre_compute: buf,
                    }
                } else {
                    return Err(ExecutionError::DisabledOperation {
                        pc,
                        opcode: inst.opcode,
                    });
                }
            } else {
                PreComputeInstruction {
                    handler: |_, vm_state| {
                        vm_state.exit_code = Err(ExecutionError::InvalidInstruction(vm_state.pc));
                    },
                    pre_compute: buf,
                }
            };
            Ok(pre_inst)
        })
        .collect::<Result<Vec<_>, _>>()
}

fn get_e2_pre_compute_instructions<
    'a,
    F: PrimeField32,
    E: InsExecutorE2<F>,
    Ctx: E2ExecutionCtx,
>(
    program: &'a Program<F>,
    inventory: &'a ExecutorInventory<E>,
    executor_idx_to_air_idx: &'a [usize],
    pre_compute: &'a mut [&mut [u8]],
) -> Result<Vec<PreComputeInstruction<'a, F, Ctx>>, ExecutionError> {
    program
        .instructions_and_debug_infos
        .iter()
        .zip_eq(pre_compute.iter_mut())
        .enumerate()
        .map(|(i, (inst_opt, buf))| {
            let buf: &mut [u8] = buf;
            let pre_inst = if let Some((inst, _)) = inst_opt {
                tracing::trace!("get_e2_pre_compute_instruction {inst:?}");
                let pc = program.pc_base + i as u32 * DEFAULT_PC_STEP;
                if let Some(handler) = get_system_opcode_handler(inst, buf) {
                    PreComputeInstruction {
                        handler,
                        pre_compute: buf,
                    }
                } else if let Some(&executor_idx) = inventory.instruction_lookup.get(&inst.opcode) {
                    let executor_idx = executor_idx as usize;
                    let executor = inventory
                        .executors
                        .get(executor_idx)
                        .expect("ExecutorInventory ensures executor_idx is in bounds");
                    let air_idx = executor_idx_to_air_idx[executor_idx];
                    PreComputeInstruction {
                        handler: executor.pre_compute_e2(air_idx, pc, inst, buf)?,
                        pre_compute: buf,
                    }
                } else {
                    return Err(ExecutionError::DisabledOperation {
                        pc,
                        opcode: inst.opcode,
                    });
                }
            } else {
                PreComputeInstruction {
                    handler: |_, vm_state| {
                        vm_state.exit_code = Err(ExecutionError::InvalidInstruction(vm_state.pc));
                    },
                    pre_compute: buf,
                }
            };
            Ok(pre_inst)
        })
        .collect::<Result<Vec<_>, _>>()
}

fn get_system_opcode_handler<F: PrimeField32, Ctx: E1ExecutionCtx>(
    inst: &Instruction<F>,
    buf: &mut [u8],
) -> Option<ExecuteFunc<F, Ctx>> {
    if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
        let pre_compute: &mut TerminatePreCompute = buf.borrow_mut();
        pre_compute.exit_code = inst.c.as_canonical_u32();
        return Some(terminate_execute_e12_impl);
    }
    None
}

fn check_exit_code<F: PrimeField32, Ctx>(
    vm_state: &VmSegmentState<F, GuestMemory, Ctx>,
) -> Result<(), ExecutionError> {
    if let Ok(Some(exit_code)) = vm_state.exit_code.as_ref() {
        if *exit_code != ExitCode::Success as u32 {
            return Err(ExecutionError::FailedWithExitCode(*exit_code));
        }
    }
    Ok(())
}
