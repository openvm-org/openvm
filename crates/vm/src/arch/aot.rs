#![cfg(feature = "aot")]
use std::{ffi::c_void, process::Command};

use libloading::Library;
use openvm_instructions::exe::{SparseMemoryImage, VmExe};
use openvm_stark_backend::p3_field::PrimeField32;

use crate::{
    arch::{
        execution_mode::{ExecutionCtx, MeteredCostCtx, MeteredCtx, Segment},
        interpreter::{
            alloc_pre_compute_buf, get_metered_pre_compute_instructions,
            get_metered_pre_compute_max_size, get_pre_compute_instructions,
            get_pre_compute_max_size, split_pre_compute_buf, AlignedBuf, PreComputeInstruction,
        },
        ExecutionCtxTrait, ExecutionError, Executor, ExecutorInventory, ExitCode,
        MeteredExecutionCtxTrait, MeteredExecutor, StaticProgramError, Streams, SystemConfig,
        VmExecState, VmState,
    },
    system::memory::online::GuestMemory,
};

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
    from_state_instret: u64,
);

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: ExecutionCtxTrait,
{
    /// Creates a new instance for pure execution
    pub fn new<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
    ) -> Result<Self, StaticProgramError>
    where
        E: Executor<F>,
    {
        let default_name = String::from("asm_x86_run");
        Self::new_with_asm_name(inventory, exe, &default_name)
    }

    /// Creates a new instance for pure execution
    /// Specify the name of the asm file
    pub fn new_with_asm_name<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        asm_name: &String, // name of the asm file we write into
    ) -> Result<Self, StaticProgramError>
    where
        E: Executor<F>,
    {
        // source asm_bridge directory
        // this is fixed
        // can unwrap because its fixed and guaranteed to exist
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root_dir = std::path::Path::new(manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap();

        let src_asm_bridge_dir = std::path::Path::new(manifest_dir).join("src/arch/asm_bridge");
        let src_asm_bridge_dir_str = src_asm_bridge_dir.to_str().unwrap();

        // ar rcs libasm_runtime.a asm_run.o
        // cargo rustc -- -L /home/ubuntu/openvm/crates/vm/src/arch/asm_bridge -l static=asm_runtime

        // run the below command from the `src_asm_bridge_dir` directory
        // as src/asm_run.s -o asm_run.o
        let status = Command::new("as")
            .current_dir(&src_asm_bridge_dir)
            .args([
                &format!("src/{}.s", asm_name),
                "-o",
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Failed to assemble the file into an object file");

        assert!(
            status.success(),
            "as src/<asm_name>.s -o <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("ar")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rcs",
                &format!("lib{}.a", asm_name),
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Create a static library");

        assert!(
            status.success(),
            "ar rcs lib<asm_name>.a <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        // library goes to `workspace_dir/target/{asm_name}/release/libasm_bridge.so`

        let status = Command::new("cargo")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rustc",
                "--release",
                &format!(
                    "--target-dir={}/target/{}",
                    root_dir.to_str().unwrap(),
                    asm_name
                ),
                "--",
                "-L",
                src_asm_bridge_dir_str,
                "-l",
                &format!("static={}", asm_name),
            ])
            .status()
            .expect("Creating the dynamic library");

        assert!(
            status.success(),
            "Cargo build failed with exit code: {:?}",
            status.code()
        );

        let lib_path = root_dir
            .join("target")
            .join(asm_name)
            .join("release")
            .join("libasm_bridge.so");

        let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };

        let program = &exe.program;
        let pre_compute_max_size = get_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);
        let pre_compute_insns = get_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            &mut split_pre_compute_buf,
        )?;
        let pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]> =
            pre_compute_insns.into_boxed_slice();

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            system_config: inventory.config().clone(),
            pre_compute_buf,
            pre_compute_insns_box,
            pc_start: exe.pc_start,
            init_memory,
            lib,
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
        &self,
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
        &self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();
        let ctx = ExecutionCtx::new(num_insns);

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, ExecutionCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, ExecutionCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns_box.as_ptr();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                from_state_instret,
            );
        }

        if num_insns.is_some() {
            check_exit_code(vm_exec_state.exit_code)?;
        } else {
            check_termination(vm_exec_state.exit_code)?;
        }

        Ok(vm_exec_state.vm_state)
    }
}

/// Errors if exit code is either error or terminated with non-successful exit code.
fn check_exit_code(exit_code: Result<Option<u32>, ExecutionError>) -> Result<(), ExecutionError> {
    let exit_code = exit_code?;
    if let Some(exit_code) = exit_code {
        // This means execution did terminate
        if exit_code != ExitCode::Success as u32 {
            return Err(ExecutionError::FailedWithExitCode(exit_code));
        }
    }
    Ok(())
}

/// Same as [check_exit_code] but errors if program did not terminate.
fn check_termination(exit_code: Result<Option<u32>, ExecutionError>) -> Result<(), ExecutionError> {
    let did_terminate = matches!(exit_code.as_ref(), Ok(Some(_)));
    check_exit_code(exit_code)?;
    match did_terminate {
        true => Ok(()),
        false => Err(ExecutionError::DidNotTerminate),
    }
}

impl<'a, F, Ctx> AotInstance<'a, F, Ctx>
where
    F: PrimeField32,
    Ctx: MeteredExecutionCtxTrait,
{
    /// Creates a new instance for metered execution.
    pub fn new_metered<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        let default_name = String::from("asm_x86_run");
        let src_asm_bridge_dir_str = String::from("src/arch/asm_bridge_metered");
        let asm_so_str = String::from("libasm_bridge_metered.so");
        Self::new_metered_generic_with_asm_name(
            inventory,
            exe,
            executor_idx_to_air_idx,
            &default_name,
            &src_asm_bridge_dir_str,
            &asm_so_str,
        )
    }

    pub fn new_metered_cost<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        let default_name = String::from("asm_x86_run");
        let src_asm_bridge_dir_str = String::from("src/arch/asm_bridge_metered_cost");
        let asm_so_str = String::from("libasm_bridge_metered_cost.so");
        Self::new_metered_generic_with_asm_name(
            inventory,
            exe,
            executor_idx_to_air_idx,
            &default_name,
            &src_asm_bridge_dir_str,
            &asm_so_str,
        )
    }

    /// Creates a new interpreter instance for metered cost execution.
    pub fn new_metered_generic_with_asm_name<E>(
        inventory: &'a ExecutorInventory<E>,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        asm_name: &String,
        bridge_str: &String,
        asm_so_str: &String,
    ) -> Result<Self, StaticProgramError>
    where
        E: MeteredExecutor<F>,
    {
        // source asm_bridge directory
        // this is fixed
        // can unwrap because its fixed and guaranteed to exist
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let root_dir = std::path::Path::new(manifest_dir)
            .parent()
            .unwrap()
            .parent()
            .unwrap();

        let src_asm_bridge_dir = std::path::Path::new(manifest_dir).join(bridge_str);
        let src_asm_bridge_dir_str = src_asm_bridge_dir.to_str().unwrap();

        // ar rcs libasm_runtime.a asm_run.o
        // cargo rustc -- -L /home/ubuntu/openvm/crates/vm/<bridge_str> -l static=asm_runtime

        // run the below command from the `src_asm_bridge_dir` directory
        // as src/asm_run.s -o asm_run.o
        let status = Command::new("as")
            .current_dir(&src_asm_bridge_dir)
            .args([
                &format!("src/{}.s", asm_name),
                "-o",
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Failed to assemble the file into an object file");

        assert!(
            status.success(),
            "as src/<asm_name>.s -o <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("ar")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rcs",
                &format!("lib{}.a", asm_name),
                &format!("{}.o", asm_name),
            ])
            .status()
            .expect("Create a static library");

        assert!(
            status.success(),
            "ar rcs lib<asm_name>.a <asm_name>.o failed with exit code: {:?}",
            status.code()
        );

        let status = Command::new("cargo")
            .current_dir(&src_asm_bridge_dir)
            .args([
                "rustc",
                "--release",
                &format!(
                    "--target-dir={}/target/{}",
                    root_dir.to_str().unwrap(),
                    asm_name
                ),
                "--",
                "-L",
                src_asm_bridge_dir_str,
                "-l",
                &format!("static={}", asm_name),
            ])
            .status()
            .expect("Creating the dynamic library");

        assert!(
            status.success(),
            "Cargo build failed with exit code: {:?}",
            status.code()
        );

        let lib_path = root_dir
            .join("target")
            .join(asm_name)
            .join("release")
            .join(asm_so_str);
        let lib = unsafe { Library::new(&lib_path).expect("Failed to load library") };

        let program = &exe.program;
        let pre_compute_max_size = get_metered_pre_compute_max_size(program, inventory);
        let mut pre_compute_buf = alloc_pre_compute_buf(program, pre_compute_max_size);
        let mut split_pre_compute_buf =
            split_pre_compute_buf(program, &mut pre_compute_buf, pre_compute_max_size);

        let pre_compute_insns = get_metered_pre_compute_instructions::<F, Ctx, E>(
            program,
            inventory,
            executor_idx_to_air_idx,
            &mut split_pre_compute_buf,
        )?;
        let pre_compute_insns_box: Box<[PreComputeInstruction<'a, F, Ctx>]> =
            pre_compute_insns.into_boxed_slice();

        let init_memory = exe.init_memory.clone();

        Ok(Self {
            system_config: inventory.config().clone(),
            pre_compute_buf,
            pre_compute_insns_box,
            pc_start: exe.pc_start,
            init_memory,
            lib,
        })
    }
}

impl<F> AotInstance<'_, F, MeteredCtx>
where
    F: PrimeField32,
{
    /// Metered execution for the given `inputs`. Execution begins from the initial
    /// state specified by the `VmExe`. This function executes the program until termination.
    ///
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    ///
    /// Assumes the program doesn't jump to out of bounds pc
    pub fn execute_metered(
        &self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_from_state(vm_state, ctx)
    }

    /// Metered execution for the given `VmState`. This function executes the program until
    /// termination
    ///
    /// Returns the segmentation boundary data and the final VM state when execution stops.
    ///
    /// Assume program doesn't jump to out of bounds pc
    pub fn execute_metered_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCtx,
    ) -> Result<(Vec<Segment>, VmState<F, GuestMemory>), ExecutionError> {
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, MeteredCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, MeteredCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns_box.as_ptr();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                from_state_instret,
            );
        }

        // handle execution error
        match vm_exec_state.exit_code {
            Ok(_) => Ok((
                vm_exec_state.ctx.segmentation_ctx.segments,
                vm_exec_state.vm_state,
            )),
            Err(e) => Err(e),
        }
    }

    // TODO: implement execute_metered_until_suspend for AOT if needed
}

impl<F> AotInstance<'_, F, MeteredCostCtx>
where
    F: PrimeField32,
{
    /// Metered cost execution for the given `inputs`. Execution begins from the initial
    /// state specified by the `VmExe`. This function executes the program until termination.
    ///
    /// Returns the cost and the final VM state when execution stops.
    ///
    /// Assumes the program doesn't jump to out of bounds pc
    pub fn execute_metered_cost(
        &mut self,
        inputs: impl Into<Streams<F>>,
        ctx: MeteredCostCtx,
    ) -> Result<(u64, VmState<F, GuestMemory>), ExecutionError> {
        let vm_state = self.create_initial_vm_state(inputs);
        self.execute_metered_cost_from_state(vm_state, ctx)
    }

    /// Metered cost execution for the given `VmState`. This function executes the program until
    /// termination
    ///
    /// Returns the cost and the final VM state when execution stops.
    ///
    /// Assume program doesn't jump to out of bounds pc
    pub fn execute_metered_cost_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        ctx: MeteredCostCtx,
    ) -> Result<(u64, VmState<F, GuestMemory>), ExecutionError> {
        let from_state_instret = from_state.instret();
        let from_state_pc = from_state.pc();

        let mut vm_exec_state: Box<VmExecState<F, GuestMemory, MeteredCostCtx>> =
            Box::new(VmExecState::new(from_state, ctx));

        unsafe {
            let asm_run: libloading::Symbol<AsmRunFn> = self
                .lib
                .get(b"asm_run")
                .expect("Failed to get asm_run symbol");

            let vm_exec_state_ptr =
                &mut *vm_exec_state as *mut VmExecState<F, GuestMemory, MeteredCostCtx>;
            let pre_compute_insns_ptr = self.pre_compute_insns_box.as_ptr();

            asm_run(
                vm_exec_state_ptr as *mut c_void,
                pre_compute_insns_ptr as *const c_void,
                from_state_pc,
                from_state_instret,
            );
        }

        // handle execution error
        match vm_exec_state.exit_code {
            Ok(_) => Ok((vm_exec_state.ctx.cost, vm_exec_state.vm_state)),
            Err(e) => Err(e),
        }
    }

    // TODO: implement execute_metered_cost_until_suspend for AOT if needed
}
