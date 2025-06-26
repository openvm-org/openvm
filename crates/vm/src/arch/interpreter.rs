use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    borrow::{Borrow, BorrowMut},
    ptr::NonNull,
};

use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    exe::VmExe,
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, SysPhantom, SystemOpcode,
};
use openvm_stark_backend::{
    p3_field::{Field, PrimeField32},
    p3_maybe_rayon::prelude::ParallelIterator,
};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    arch::{
        execution_control::ExecutionControl, execution_mode::E1E2ExecutionCtx, ExecutionError,
        InsExecutorE1, PreComputeInstruction, Streams, VmConfig, VmSegmentState,
    },
    next_instruction,
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
    pub fn execute<CTRL: ExecutionControl<F, VC>>(
        &self,
        ctrl: CTRL,
        inputs: impl Into<Streams<F>>,
    ) -> Result<VmSegmentState<F, CTRL::Ctx>, ExecutionError>
    where
        CTRL::Ctx: E1E2ExecutionCtx,
    {
        // Initialize the chip complex
        let mut chip_complex = self.vm_config.create_chip_complex().unwrap();
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

        // Initialize the context
        let ctx = ctrl.initialize_context();

        let mut vm_state = VmSegmentState::new(
            0,
            self.exe.pc_start,
            memory,
            inputs.into(),
            StdRng::seed_from_u64(0),
            ctx,
        );

        // Start execution
        ctrl.on_start(&mut vm_state, &mut chip_complex);
        let program = &self.exe.program;

        let pre_compute_max_size = program
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
            .next_power_of_two();
        let program_len = program.instructions_and_debug_infos.len();
        let buf_len = program_len * pre_compute_max_size;
        let pre_compute_buf = AlignedBuf::uninit(buf_len, pre_compute_max_size);
        let pre_compute_buf =
            unsafe { std::slice::from_raw_parts_mut(pre_compute_buf.ptr, buf_len) };
        program
            .instructions_and_debug_infos
            .iter()
            .enumerate()
            .for_each(|(i, inst_opt)| {
                if let Some((inst, _)) = inst_opt {
                    let buf = &mut pre_compute_buf[i * pre_compute_max_size..];
                    let discriminant = SysPhantom::from_repr(inst.c.as_canonical_u32() as u16);
                    if inst.opcode == SystemOpcode::PHANTOM.global_opcode()
                        && discriminant.is_some()
                    {
                        let discriminant = discriminant.unwrap();
                        if discriminant == SysPhantom::DebugPanic {
                            let pre_compute: &mut DebugPanicPreCompute = buf.borrow_mut();
                            pre_compute.pc =
                                self.exe.program.pc_base + i as u32 * self.exe.program.step;
                        }
                    } else if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
                        let pre_compute: &mut TerminatePreCompute = buf.borrow_mut();
                        pre_compute.exit_code = inst.c.as_canonical_u32();
                    } else {
                        let executor = chip_complex.inventory.get_executor(inst.opcode).unwrap();
                        executor.pre_compute(inst, buf);
                    }
                }
            });

        let pre_compute_insts: Vec<PreComputeInstruction<F, CTRL::Ctx>> = program
            .instructions_and_debug_infos
            .iter()
            .enumerate()
            .map(|(i, inst_opt)| {
                let buf = &pre_compute_buf[i * pre_compute_max_size..];
                if let Some((inst, _)) = inst_opt {
                    let discriminant = SysPhantom::from_repr(inst.c.as_canonical_u32() as u16);
                    if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
                        PreComputeInstruction {
                            handler: terminate_execute_e1_impl,
                            pre_compute: buf,
                        }
                    } else if inst.opcode == SystemOpcode::PHANTOM.global_opcode()
                        && discriminant.is_some()
                    {
                        let discriminant = discriminant.unwrap();
                        PreComputeInstruction {
                            handler: match discriminant {
                                SysPhantom::Nop => nop_execute_e1_impl,
                                SysPhantom::DebugPanic => debug_panic_execute_e1_impl,
                                SysPhantom::CtStart => nop_execute_e1_impl,
                                SysPhantom::CtEnd => nop_execute_e1_impl,
                            },
                            pre_compute: buf,
                        }
                    } else {
                        let executor = chip_complex.inventory.get_executor(inst.opcode).unwrap();
                        PreComputeInstruction {
                            handler: executor.execute_e1(),
                            pre_compute: buf,
                        }
                    }
                } else {
                    PreComputeInstruction {
                        handler: |_, _| panic!("Empty instruction!"),
                        pre_compute: buf,
                    }
                }
            })
            .collect();

        // while vm_state.exit_code.is_none() {
        //     let pc_index = get_pc_index(program, vm_state.pc)?;
        //     let inst = &pre_compute_insts[pc_index];
        //     unsafe { (inst.handler)(inst, &mut vm_state) };
        // }
        let start = std::time::Instant::now();
        unsafe {
            execute_impl(program, &mut vm_state, &pre_compute_insts);
        }
        println!("time {}ms", start.elapsed().as_millis());
        // let start_pc_index = get_pc_index(program, self.exe.pc_start)?;
        // let start_inst = &pre_compute_insts[start_pc_index];
        // unsafe { start_e1(start_inst, &mut vm_state) };
        if let Some(exit_code) = vm_state.exit_code {
            ctrl.on_terminate(&mut vm_state, &mut chip_complex, exit_code);
        } else {
            panic!("Execution did not terminate");
        }
        Ok(vm_state)

        // loop {
        //     if ctrl.should_suspend(&mut vm_state, &chip_complex) {
        //         ctrl.on_suspend(&mut vm_state, &mut chip_complex);
        //     }
        //
        //     // Fetch the next instruction
        //     let pc_index = get_pc_index(program, vm_state.pc)?;
        //     debug_assert!(pc_index < program_len);
        //     let buf = &pre_compute_buf[pc_index * pre_compute_max_size..];
        //
        //     let (inst, _) = program.get_instruction_and_debug_info(pc_index).ok_or(
        //         ExecutionError::PcNotFound {
        //             pc,
        //             step: program.step,
        //             pc_base: program.pc_base,
        //             program_len: program.len(),
        //         },
        //     )?;
        //     if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
        //         let exit_code = inst.c.as_canonical_u32();
        //         vm_state.exit_code = Some(exit_code);
        //         ctrl.on_terminate(&mut vm_state, &mut chip_complex, exit_code);
        //         return Ok(vm_state);
        //     }
        //     ctrl.execute_instruction(&mut vm_state, inst, &mut chip_complex)?;
        // }
    }
}

#[inline(never)]
unsafe fn execute_impl<F: PrimeField32, Ctx: E1E2ExecutionCtx>(
    program: &Program<F>,
    vm_state: &mut VmSegmentState<F, Ctx>,
    pre_compute_insts: &[PreComputeInstruction<F, Ctx>],
) {
    while vm_state.exit_code.is_none() {
        let pc_index = get_pc_index(program, vm_state.pc).unwrap();
        let inst = &pre_compute_insts[pc_index];
        unsafe { (inst.handler)(inst, vm_state) };
    }
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

#[inline(never)]
unsafe fn start_e1<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    next_instruction!(inst, vm_state);
}

unsafe fn terminate_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    // println!("Terminating VM execution");
    // panic!();
    let inst = &*inst;
    let pre_compute: &TerminatePreCompute = inst.pre_compute.borrow();
    vm_state.exit_code = Some(pre_compute.exit_code);
    // Ok(())
}

unsafe fn debug_panic_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    _vm_state: &mut VmSegmentState<F, CTX>,
) {
    let inst = unsafe { &*inst };
    let pre_compute: &DebugPanicPreCompute = inst.pre_compute.borrow();
    // Err(ExecutionError::Fail { pc: pre_compute.pc })
}

unsafe fn nop_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) {
    let next_inst = unsafe { inst.offset(1) };
    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
    next_instruction!(next_inst, vm_state)
}
