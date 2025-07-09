use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    borrow::{Borrow, BorrowMut},
    ptr::NonNull,
    time::Instant,
};

use itertools::Itertools;
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    exe::VmExe,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SysPhantom, SystemOpcode,
};
use openvm_stark_backend::p3_field::{Field, PrimeField32};
use rand::{rngs::StdRng, SeedableRng};
use tracing::info_span;

use crate::{
    arch::{
        execution_mode::E1ExecutionCtx, ExecutionError, ExecutionError::InvalidInstruction,
        InsExecutorE1, PreComputeInstruction, Streams, VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::{online::GuestMemory, AddressMap},
};

/// VM pure executor(E1/E2 executor) which doesn't consider trace generation.
/// Note: This executor doesn't hold any VM state and can be used for multiple execution.
pub struct InterpretedInstance<F: PrimeField32, VC: VmConfig<F>> {
    exe: VmExe<F>,
    vm_config: VC,
}

#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct TerminatePreCompute {
    exit_code: u32,
}
#[derive(AlignedBytesBorrow, Clone)]
#[repr(C)]
struct DebugPanicPreCompute {
    pc: u32,
}

impl<F: PrimeField32, VC: VmConfig<F>> InterpretedInstance<F, VC> {
    pub fn new(vm_config: VC, exe: impl Into<VmExe<F>>) -> Self {
        let exe = exe.into();
        Self { exe, vm_config }
    }

    /// Execute the VM program with the given execution control and inputs. Returns the final VM
    /// state with the `ExecutionControl` context.
    pub fn execute<Ctx: E1ExecutionCtx>(
        &self,
        ctx: Ctx,
        inputs: impl Into<Streams<F>>,
    ) -> Result<VmSegmentState<F, Ctx>, ExecutionError> {
        // Initialize the chip complex
        let chip_complex = self.vm_config.create_chip_complex().unwrap();
        // Initialize the memory
        let memory = if self.vm_config.system().continuation_enabled {
            let mem_config = self.vm_config.system().memory_config.clone();
            Some(GuestMemory::new(AddressMap::from_sparse(
                mem_config.addr_space_sizes.clone(),
                self.exe.init_memory.clone(),
            )))
        } else {
            None
        };

        let mut vm_state = VmSegmentState::new(
            0,
            self.exe.pc_start,
            memory,
            inputs.into(),
            StdRng::seed_from_u64(0),
            ctx,
        );

        // Start execution
        let program = &self.exe.program;
        let pre_compute_max_size = get_pre_compute_max_size(program, &chip_complex);
        let program_len = program.instructions_and_debug_infos.len();
        let buf_len = program_len * pre_compute_max_size;
        let pre_compute_buf = AlignedBuf::uninit(buf_len, pre_compute_max_size);
        let mut pre_compute_buf =
            unsafe { std::slice::from_raw_parts_mut(pre_compute_buf.ptr, buf_len) };
        let mut split_pre_compute_buf = Vec::with_capacity(program_len);
        for _ in 0..program_len {
            let (first, last) = pre_compute_buf.split_at_mut(pre_compute_max_size);
            pre_compute_buf = last;
            split_pre_compute_buf.push(first);
        }

        let pre_compute_insts = get_pre_compute_instructions::<_, _, _, Ctx>(
            program,
            &chip_complex,
            &mut split_pre_compute_buf,
        )?;

        execute_with_metrics(program, &mut vm_state, &pre_compute_insts);

        if vm_state.exit_code.is_err() {
            Err(vm_state.exit_code.err().unwrap())
        } else {
            Ok(vm_state)
        }
    }
}

fn execute_with_metrics<F: PrimeField32, Ctx: E1ExecutionCtx>(
    program: &Program<F>,
    vm_state: &mut VmSegmentState<F, Ctx>,
    pre_compute_insts: &[PreComputeInstruction<F, Ctx>],
) {
    #[cfg(feature = "bench-metrics")]
    let start = std::time::Instant::now();
    #[cfg(feature = "bench-metrics")]
    let start_instret = vm_state.instret;

    info_span!("execute_e1").in_scope(|| unsafe {
        execute_impl(program, vm_state, pre_compute_insts);
    });

    #[cfg(feature = "bench-metrics")]
    {
        let elapsed = start.elapsed();
        let insns = vm_state.instret - start_instret;
        metrics::counter!("insns").absolute(insns);
        metrics::gauge!(concat!("execute_e1", "_insn_mi/s"))
            .set(insns as f64 / elapsed.as_micros() as f64);
    }
}

#[inline(never)]
unsafe fn execute_impl<F: PrimeField32, Ctx: E1ExecutionCtx>(
    program: &Program<F>,
    vm_state: &mut VmSegmentState<F, Ctx>,
    fn_ptrs: &[PreComputeInstruction<F, Ctx>],
) {
    let start = Instant::now();
    while vm_state
        .exit_code
        .as_ref()
        .is_ok_and(|exit_code| exit_code.is_none())
    {
        let pc_index = get_pc_index(program, vm_state.pc).unwrap();
        let inst = &fn_ptrs[pc_index];
        unsafe { (inst.handler)(inst.pre_compute, vm_state) };
        if Ctx::should_suspend(vm_state) {
            break;
        }
    }
    println!("execute time: {}ms", start.elapsed().as_millis());
}

fn get_pc_index<F: Field>(program: &Program<F>, pc: u32) -> Result<usize, ExecutionError> {
    let step = program.step;
    let pc_base = program.pc_base;
    let pc_index = ((pc - pc_base) / step) as usize;
    if !(0..program.len()).contains(&pc_index) {
        return Err(ExecutionError::PcOutOfBounds {
            pc,
            step,
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

unsafe fn terminate_execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &TerminatePreCompute = pre_compute.borrow();
    vm_state.exit_code = Ok(Some(pre_compute.exit_code));
}

unsafe fn debug_panic_execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let pre_compute: &DebugPanicPreCompute = pre_compute.borrow();
    vm_state.exit_code = Err(ExecutionError::Fail { pc: pre_compute.pc });
}

unsafe fn nop_execute_e1_impl<F: PrimeField32, CTX: E1ExecutionCtx>(
    _pre_compute: &[u8],
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
}

fn get_pre_compute_max_size<F: PrimeField32, E: InsExecutorE1<F>, P>(
    program: &Program<F>,
    chip_complex: &VmChipComplex<F, E, P>,
) -> usize {
    program
        .instructions_and_debug_infos
        .iter()
        .map(|inst_opt| {
            if let Some((inst, _)) = inst_opt {
                let discriminant = SysPhantom::from_repr(inst.c.as_canonical_u32() as u16);
                if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
                    size_of::<TerminatePreCompute>()
                } else if inst.opcode == SystemOpcode::PHANTOM.global_opcode()
                    && discriminant.is_some()
                {
                    let discriminant = discriminant.unwrap();
                    match discriminant {
                        SysPhantom::DebugPanic => size_of::<DebugPanicPreCompute>(),
                        SysPhantom::Nop | SysPhantom::CtStart | SysPhantom::CtEnd => 0,
                    }
                } else {
                    chip_complex
                        .inventory
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

fn get_pre_compute_instructions<
    'a,
    F: PrimeField32,
    E: InsExecutorE1<F>,
    P,
    Ctx: E1ExecutionCtx,
>(
    program: &'a Program<F>,
    chip_complex: &'a VmChipComplex<F, E, P>,
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
                let pc = program.pc_base + i as u32 * program.step;
                let discriminant = SysPhantom::from_repr(inst.c.as_canonical_u32() as u16);
                if inst.opcode == SystemOpcode::PHANTOM.global_opcode() && discriminant.is_some() {
                    let discriminant = discriminant.unwrap();
                    if discriminant == SysPhantom::DebugPanic {
                        let pre_compute: &mut DebugPanicPreCompute = buf.borrow_mut();
                        pre_compute.pc = pc;
                    }
                    PreComputeInstruction {
                        handler: match discriminant {
                            SysPhantom::Nop => nop_execute_e1_impl,
                            SysPhantom::DebugPanic => debug_panic_execute_e1_impl,
                            SysPhantom::CtStart => nop_execute_e1_impl,
                            SysPhantom::CtEnd => nop_execute_e1_impl,
                        },
                        pre_compute: buf,
                    }
                } else if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
                    let pre_compute: &mut TerminatePreCompute = buf.borrow_mut();
                    pre_compute.exit_code = inst.c.as_canonical_u32();
                    PreComputeInstruction {
                        handler: terminate_execute_e1_impl,
                        pre_compute: buf,
                    }
                } else if let Some(executor) = chip_complex.inventory.get_executor(inst.opcode) {
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
                        vm_state.exit_code = Err(InvalidInstruction(vm_state.pc));
                    },
                    pre_compute: buf,
                }
            };
            Ok(pre_inst)
        })
        .collect::<Result<Vec<_>, _>>()
}
