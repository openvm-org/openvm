use openvm_instructions::{exe::VmExe, program::Program, LocalOpcode, SystemOpcode};
use openvm_stark_backend::p3_field::{Field, PrimeField32};

use crate::{
    arch::{
        execution_control::ExecutionControl, execution_mode::E1E2ExecutionCtx, ExecutionError,
        Streams, VmChipComplex, VmConfig, VmSegmentState,
    },
    system::memory::{online::GuestMemory, AddressMap},
};

/// VM pure executor(E1/E2 executor) which doesn't consider trace generation.
/// Note: This executor doesn't hold any VM state and can be used for multiple execution.
pub struct InterpretedInstance<F: PrimeField32, VC: VmConfig<F>> {
    exe: VmExe<F>,
    vm_config: VC,
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
    ) -> Result<VmSegmentState<CTRL::Ctx>, ExecutionError>
    where
        CTRL::Ctx: E1E2ExecutionCtx,
    {
        // Initialize the chip complex
        let mut chip_complex = self.vm_config.create_chip_complex().unwrap();
        let inputs = inputs.into();
        chip_complex.set_streams(inputs);
        // Initialize the memory
        let memory = if self.vm_config.system().continuation_enabled {
            let mem_config = self.vm_config.system().memory_config;
            Some(GuestMemory::new(AddressMap::from_sparse(
                mem_config.as_offset,
                1 << mem_config.as_height,
                1 << mem_config.pointer_max_bits,
                self.exe.init_memory.clone(),
            )))
        } else {
            Some(GuestMemory::new(Default::default()))
        };

        // Initialize the context
        let ctx = ctrl.initialize_context();
        let mut vm_state = VmSegmentState {
            clk: 0,
            pc: self.exe.pc_start,
            memory,
            exit_code: None,
            ctx,
        };

        // Start execution
        ctrl.on_start(&mut vm_state, &mut chip_complex);
        let program = &self.exe.program;

        loop {
            if ctrl.should_suspend(&mut vm_state, &chip_complex) {
                ctrl.on_suspend(&mut vm_state, &mut chip_complex);
            }

            // Fetch the next instruction
            let pc = vm_state.pc;
            let pc_index = get_pc_index(program, vm_state.pc)?;
            let (inst, _) = program.get_instruction_and_debug_info(pc_index).ok_or(
                ExecutionError::PcNotFound {
                    pc,
                    step: program.step,
                    pc_base: program.pc_base,
                    program_len: program.len(),
                },
            )?;
            if inst.opcode == SystemOpcode::TERMINATE.global_opcode() {
                let exit_code = inst.c.as_canonical_u32();
                vm_state.exit_code = Some(exit_code);
                ctrl.on_terminate(&mut vm_state, &mut chip_complex, exit_code);
                return Ok(vm_state);
            }
            ctrl.execute_instruction(&mut vm_state, inst, &mut chip_complex)?;
        }
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
