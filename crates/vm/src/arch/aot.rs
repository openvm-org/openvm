use openvm_instructions::exe::VmExe;
use crate::arch::VmExecState;
use crate::arch::VmState; 
use openvm_instructions::program::Program;
use openvm_instructions::instruction::Instruction;
use openvm_rv32im_transpiler::{
    BaseAluOpcode,
    BranchEqualOpcode
};
use openvm_instructions::LocalOpcode;
use crate::arch::MemoryConfig;
use p3_baby_bear::BabyBear;
use openvm_stark_backend::p3_field::PrimeField32;
use crate::arch::SystemConfig;
use crate::system::memory::online::GuestMemory;
use openvm_instructions::exe::SparseMemoryImage;
use crate::arch::execution_mode::ExecutionCtx;
use crate::arch::interpreter::{get_pre_compute_max_size, alloc_pre_compute_buf, split_pre_compute_buf, get_pre_compute_instructions, get_metered_pre_compute_max_size, get_metered_pre_compute_instructions};
use crate::arch::ExecutorInventory;
use crate::arch::VmExecutor;
use crate::derive::VmConfig;
use strum::{EnumCount, EnumIter, FromRepr, IntoEnumIterator};
use std::process::Command;
use memmap2::MmapOptions;
use crate::arch::StaticProgramError;
use crate::arch::interpreter::AlignedBuf;
use std::fs;
use memmap2::MmapMut;
use crate::arch::Streams;
use std::ffi::c_void;
use libloading::{Library, Symbol};
use crate::arch::{
    interpreter::PreComputeInstruction,
    ExecutionError,
};
use crate::arch::Executor;

use openvm_stark_backend::config::Val;
use crate::arch::ExecutionCtxTrait;
use crate::arch::InterpretedInstance;
use crate::arch::execution_mode::MeteredCtx;
use crate::arch::execution_mode::Segment;
use crate::arch::instructions::SystemOpcode::TERMINATE;
use crate::arch::MeteredExecutor;
use crate::arch::MeteredExecutionCtxTrait;

pub struct AotInstance<'a, F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    pre_compute_buf: AlignedBuf,
    lib: Library,
    pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]>,
    pc_base: u32,
    pc_start: u32,
}

use std::sync::Mutex;
use std::thread;

type AsmRunFn = unsafe extern "C" fn(
    exec_state: *mut c_void, 
    vec_ptr: *const c_void,
    pc: u32,
    instret: u64,
    pc_base: u32
);

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where 
    F: PrimeField32,
    Ctx: ExecutionCtxTrait
{
    pub fn new<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError> 
    where
        E: Executor<F>,
    {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let lib_path = std::path::Path::new(manifest_dir)
            .parent().unwrap()
            .parent().unwrap()
            .join("target/release/libasm_bridge.so");
        let asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&asm_bridge_dir)
            .status()
            .expect("Failed to execute cargo");
        assert!(status.success(), "Cargo build failed with exit code: {:?}", status.code());

        let lib = unsafe {
            Library::new(&lib_path).expect("Failed to load library")
        };

        let program = &exe.program; 
        let pre_compute_max_size = get_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;
        let pre_compute_insns_box : Box<[PreComputeInstruction<'a, F, Ctx>]> = pre_compute_insns.into_boxed_slice();

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            pre_compute_buf: pre_compute_buf,
            system_config: inventory.config().clone(),
            init_memory: init_memory,
            lib: lib,
            pre_compute_insns_box: pre_compute_insns_box,
            pc_base: program.pc_base,
            pc_start: exe.pc_start,
        })
    }   

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams<F>>) -> VmState<F> {
        VmState::initial(
            &self.system_config,
            &self.init_memory,
            self.pc_start,
            inputs,
        )
    }
}

impl<F> AotInstance<'_, F, ExecutionCtx>
where 
    F: PrimeField32,
{
    /// Pure AOT execution, without metering, for the given `inputs`.
    /// this function executes the program until termination
    /// Returns the final VM state when execution stops.
    pub fn execute(
        &mut self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let vm_state = VmState::initial(
            &self.system_config,
            &self.init_memory,
            self.pc_start,
            inputs,
        );
        self.execute_from_state(vm_state, num_insns)
    }

    // Runs pure execution with AOT starting with `from_state` VmState
    // Runs for `num_insns` instructions if `num_insns` is not None
    // Otherwise executes until termination
    pub fn execute_from_state(
        &mut self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        type Ctx = ExecutionCtx;

        let from_state_instret = (&from_state).instret();
        let from_state_pc = (&from_state).pc();
        let ctx = ExecutionCtx::new(num_insns);

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, Ctx>> = Box::new(VmExecState::new(from_state, ctx));
        
        println!("from_state_instret {}", from_state_instret);
        println!("from_state_pc {}", from_state_pc);

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self.lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            
            let vm_exec_state_ptr = &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, Ctx>;
            let pre_compute_insns_ptr = (&self.pre_compute_insns_box).as_ptr();
            let pc_base = self.pc_base;

            asm_run(
                vm_exec_state_ptr as *mut c_void, 
                pre_compute_insns_ptr as *const c_void, 
                from_state_pc,
                from_state_instret,
                pc_base
            );
        }

        Ok((*vm_exec_state).vm_state)
    }
}




impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where 
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait
{
    pub fn new_metered<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError> 
    where
        E: MeteredExecutor<F>,
    {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let lib_path = std::path::Path::new(manifest_dir)
            .parent().unwrap()
            .parent().unwrap()
            .join("target/release/libasm_bridge_metered.so");
        let asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge_metered");
        let status = Command::new("cargo")
            .args(&["build", "--release"])
            .current_dir(&asm_bridge_dir)
            .status()
            .expect("Failed to execute cargo");
        assert!(status.success(), "Cargo build failed with exit code: {:?}", status.code());

        let lib = unsafe {
            Library::new(&lib_path).expect("Failed to load library")
        };

        let program = &exe.program; 
        let pre_compute_max_size = get_metered_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf = split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insns = get_metered_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            executor_idx_to_air_idx,
            &mut split_pre_compute_buf,
        )?;
        let pre_compute_insns_box : Box<[PreComputeInstruction<'a, F, Ctx>]> = pre_compute_insns.into_boxed_slice();

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            pre_compute_buf: pre_compute_buf,
            system_config: inventory.config().clone(),
            init_memory: init_memory,
            lib: lib,
            pre_compute_insns_box: pre_compute_insns_box,
            pc_base: program.pc_base,
            pc_start: exe.pc_start,
        })
    }   
}

impl<F> AotInstance<'_, F, MeteredCtx>
where 
    F: PrimeField32,
{
    /// Metered exeecution for the given `inputs`. Execution begins from the initial
    /// state specified by the `VmExe`. This function executes the program until termination.
    /// 
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    pub fn execute_metered(
        &mut self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    /// Metered execution for the given `VmState`. This function executes the program until 
    /// termination
    /// 
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    pub fn execute_metered_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        type Ctx = MeteredCtx;

        let from_state_instret = (&from_state).instret();
        let from_state_pc = (&from_state).pc();

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, Ctx>> = Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self.lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");
            
            let vm_exec_state_ptr = &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, Ctx>;
            let pre_compute_insns_ptr = (&self.pre_compute_insns_box).as_ptr();
            let pc_base = self.pc_base;

            asm_run(
                vm_exec_state_ptr as *mut c_void, 
                pre_compute_insns_ptr as *const c_void, 
                from_state_pc,
                from_state_instret,
                pc_base
            );
        }

        Ok(((*vm_exec_state).ctx.segmentation_ctx.segments ,(*vm_exec_state).vm_state))
    }


    // TODO: execute_metered_until_suspend ? 
}