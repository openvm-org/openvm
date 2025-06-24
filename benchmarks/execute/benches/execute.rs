use std::{path::Path, sync::OnceLock};

use divan::Bencher;
use eyre::Result;
use openvm_benchmarks_utils::{get_elf_path, get_programs_dir, read_elf_file};
use openvm_circuit::arch::execution_mode::e1::E1ExecutionControl;
// use openvm_bigint_circuit::{Int256, Int256Executor, Int256Periphery};
// use openvm_bigint_transpiler::Int256TranspilerExtension;
use openvm_circuit::{
    arch::{
        create_initial_state, instructions::exe::VmExe, InitFileGenerator, SystemConfig,
        VirtualMachine,
    },
    derive::VmConfig,
};
// use openvm_keccak256_circuit::{Keccak256, Keccak256Executor, Keccak256Periphery};
// use openvm_keccak256_transpiler::Keccak256TranspilerExtension;
use openvm_rv32im_circuit::{
    Rv32I, Rv32IExecutor, Rv32IPeriphery, Rv32Io, Rv32IoExecutor, Rv32IoPeriphery, Rv32M,
    Rv32MExecutor, Rv32MPeriphery,
};
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
// use openvm_sha256_circuit::{Sha256, Sha256Executor, Sha256Periphery};
// use openvm_sha256_transpiler::Sha256TranspilerExtension;
use openvm_stark_sdk::{
    config::baby_bear_poseidon2::{
        default_engine, BabyBearPoseidon2Config, BabyBearPoseidon2Engine,
    },
    openvm_stark_backend::{self, p3_field::PrimeField32},
    p3_baby_bear::BabyBear,
};
use openvm_transpiler::{transpiler::Transpiler, FromElf};
use serde::{Deserialize, Serialize};

static AVAILABLE_PROGRAMS: &[&str] = &[
    "fibonacci_recursive",
    "fibonacci_iterative",
    "quicksort",
    "bubblesort",
    // "factorial_iterative_u256",
    // "revm_snailtracer",
    // "keccak256",
    // "keccak256_iter",
    // "sha256",
    // "sha256_iter",
    // "revm_transfer",
    // "pairing",
];

static SHARED_WIDTHS_AND_INTERACTIONS: OnceLock<(Vec<usize>, Vec<usize>)> = OnceLock::new();

// TODO(ayush): remove from here
#[derive(Clone, Debug, VmConfig, Serialize, Deserialize)]
pub struct ExecuteConfig {
    #[system]
    pub system: SystemConfig,
    #[extension]
    pub rv32i: Rv32I,
    #[extension]
    pub rv32m: Rv32M,
    #[extension]
    pub io: Rv32Io,
    // #[extension]
    // pub bigint: Int256,
    // #[extension]
    // pub keccak: Keccak256,
    // #[extension]
    // pub sha256: Sha256,
}

impl Default for ExecuteConfig {
    fn default() -> Self {
        Self {
            system: SystemConfig::default().with_continuations(),
            rv32i: Rv32I::default(),
            rv32m: Rv32M::default(),
            io: Rv32Io::default(),
            // bigint: Int256::default(),
            // keccak: Keccak256::default(),
            // sha256: Sha256::default(),
        }
    }
}

impl InitFileGenerator for ExecuteConfig {
    fn write_to_init_file(
        &self,
        _manifest_dir: &Path,
        _init_file_name: Option<&str>,
    ) -> eyre::Result<()> {
        Ok(())
    }
}

fn main() {
    divan::main();
}

fn create_default_vm(
) -> VirtualMachine<BabyBearPoseidon2Config, BabyBearPoseidon2Engine, ExecuteConfig> {
    let vm_config = ExecuteConfig::default();
    VirtualMachine::new(default_engine(), vm_config)
}

fn create_default_transpiler() -> Transpiler<BabyBear> {
    Transpiler::<BabyBear>::default()
        .with_extension(Rv32ITranspilerExtension)
        .with_extension(Rv32IoTranspilerExtension)
        .with_extension(Rv32MTranspilerExtension)
    // .with_extension(Int256TranspilerExtension)
    // .with_extension(Keccak256TranspilerExtension)
    // .with_extension(Sha256TranspilerExtension)
}

fn load_program_executable(program: &str) -> Result<VmExe<BabyBear>> {
    let transpiler = create_default_transpiler();
    let program_dir = get_programs_dir().join(program);
    let elf_path = get_elf_path(&program_dir);
    let elf = read_elf_file(&elf_path)?;
    Ok(VmExe::from_elf(elf, transpiler)?)
}

fn shared_widths_and_interactions() -> &'static (Vec<usize>, Vec<usize>) {
    SHARED_WIDTHS_AND_INTERACTIONS.get_or_init(|| {
        let vm = create_default_vm();
        let pk = vm.keygen();
        let vk = pk.get_vk();
        (vk.total_widths(), vk.num_interactions())
    })
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=10)]
fn benchmark_execute(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let vm_config = ExecuteConfig::default();
            let interpreter = InterpretedInstance::new(vm_config, exe);
            (interpreter, vec![])
        })
        .bench_values(|(interpreter, input)| {
            interpreter
                .execute(E1ExecutionControl::new(None), input)
                .expect("Failed to execute program in interpreted mode");
        });
}

#[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=5)]
fn benchmark_execute_metered(bencher: Bencher, program: &str) {
    bencher
        .with_inputs(|| {
            let vm = create_default_vm();
            let exe = load_program_executable(program).expect("Failed to load program executable");
            let state = create_initial_state(&vm.config().system.memory_config, &exe, vec![]);

            let (widths, interactions) = shared_widths_and_interactions();
            (vm.executor, exe, state, widths, interactions)
        })
        .bench_values(|(executor, exe, state, widths, interactions)| {
            executor
                .execute_metered_from_state(exe, state, widths, interactions)
                .expect("Failed to execute program");
        });
}

// #[divan::bench(args = AVAILABLE_PROGRAMS, sample_count=3)]
// fn benchmark_execute_e3(bencher: Bencher, program: &str) {
//     bencher
//         .with_inputs(|| {
//             let vm = create_default_vm();
//             let exe = load_program_executable(program).expect("Failed to load program
// executable");             let state = create_initial_state(&vm.config().system.memory_config,
// &exe, vec![]);

//             let (widths, interactions) = shared_widths_and_interactions();
//             let segments = vm
//                 .executor
//                 .execute_metered(exe.clone(), vec![], widths, interactions)
//                 .expect("Failed to execute program");

//             (vm.executor, exe, state, segments)
//         })
//         .bench_values(|(executor, exe, state, segments)| {
//             executor
//                 .execute_from_state(exe, state, &segments)
//                 .expect("Failed to execute program");
//         });
// }

use std::{
    alloc::{alloc, dealloc, handle_alloc_error, Layout},
    borrow::{Borrow, BorrowMut},
    ptr::NonNull,
};

use openvm_circuit::{
    arch::{
        execution_control::ExecutionControl, execution_mode::E1E2ExecutionCtx,
        instructions::VmOpcode, ExecutionError, InsExecutorE1, PhantomSubExecutor,
        PreComputeInstruction, Streams, VmConfig, VmSegmentState,
    },
    next_instruction,
    system::memory::{online::GuestMemory, AddressMap},
};
use openvm_circuit_primitives_derive::AlignedBytesBorrow;
use openvm_instructions::{
    program::{Program, DEFAULT_PC_STEP},
    LocalOpcode, PhantomDiscriminant, SysPhantom, SystemOpcode,
};
use openvm_stark_backend::{
    p3_field::Field,
    p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator},
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

struct SystemSubEx;

impl<F: Field> PhantomSubExecutor<F> for SystemSubEx {
    fn phantom_execute(
        &self,
        _: &GuestMemory,
        _: &mut Streams<F>,
        discriminant: PhantomDiscriminant,
        a: u32,
        b: u32,
        c_upper: u16,
    ) -> eyre::Result<()> {
        todo!()
    }
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
            let mem_config = self.vm_config.system().memory_config;
            Some(GuestMemory::new(AddressMap::from_sparse(
                mem_config.as_offset,
                1 << mem_config.as_height,
                1 << mem_config.pointer_max_bits,
                self.exe.init_memory.clone(),
            )))
        } else {
            None
        };

        // Initialize the context
        let ctx = ctrl.initialize_context();

        let mut vm_state = VmSegmentState::new(0, self.exe.pc_start, memory, inputs.into(), ctx);

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
        let opcode_c_lists: Vec<_> = program
            .instructions_and_debug_infos
            .iter()
            .map(|inst_opt| {
                if let Some((inst, _)) = inst_opt {
                    (inst.opcode.as_usize(), inst.c.as_canonical_u32())
                } else {
                    (0, 0)
                }
            })
            .collect();

        while vm_state.exit_code.is_none() {
            let pc_index = get_pc_index(program, vm_state.pc)?;

            let inst = &pre_compute_insts[pc_index];
            unsafe {
                enum_execute_e1(
                    opcode_c_lists[pc_index].0,
                    opcode_c_lists[pc_index].1,
                    inst as *const PreComputeInstruction<F, CTRL::Ctx>,
                    &mut vm_state,
                )?;
            }
        }
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
unsafe fn terminate_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) -> openvm_circuit::arch::Result<()> {
    let inst = &*inst;
    let pre_compute: &TerminatePreCompute = inst.pre_compute.borrow();
    vm_state.exit_code = Some(pre_compute.exit_code);
    Ok(())
}

unsafe fn debug_panic_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    _vm_state: &mut VmSegmentState<F, CTX>,
) -> openvm_circuit::arch::Result<()> {
    let inst = unsafe { &*inst };
    let pre_compute: &DebugPanicPreCompute = inst.pre_compute.borrow();
    Err(ExecutionError::Fail { pc: pre_compute.pc })
}

unsafe fn nop_execute_e1_impl<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) -> openvm_circuit::arch::Result<()> {
    let next_inst = unsafe { inst.offset(1) };
    vm_state.pc += DEFAULT_PC_STEP;
    vm_state.instret += 1;
    next_instruction!(next_inst, vm_state)
}

unsafe fn enum_execute_e1<F: PrimeField32, CTX: E1E2ExecutionCtx>(
    opcode: usize,
    c: u32,
    inst: *const PreComputeInstruction<F, CTX>,
    vm_state: &mut VmSegmentState<F, CTX>,
) -> openvm_circuit::arch::Result<()> {
    use openvm_rv32im_circuit::*;
    match opcode {
        0x200 | 0x201 | 0x202 | 0x203 | 0x204 => {
            base_alu::execute_e1_impl(inst, vm_state)?;
        }
        0x205 | 0x206 | 0x207 => {
            shift::execute_e1_impl(inst, vm_state)?;
        }
        0x208 | 0x209 => {
            less_than::execute_e1_impl(inst, vm_state)?;
        }
        0x216 | 0x217 => {
            load_sign_extend::execute_e1_impl(inst, vm_state)?;
        }
        0x210 | 0x211 | 0x212 | 0x213 | 0x214 | 0x215 => {
            loadstore::execute_e1_impl(inst, vm_state)?;
        }
        0x220 | 0x221 => {
            branch_eq::execute_e1_impl(inst, vm_state)?;
        }
        0x225 | 0x226 | 0x227 | 0x228 => {
            branch_lt::execute_e1_impl(inst, vm_state)?;
        }
        0x230 | 0x231 => {
            jal_lui::execute_e1_impl(inst, vm_state)?;
        }
        0x235 => {
            jalr::execute_e1_impl(inst, vm_state)?;
        }
        0x240 => {
            auipc::execute_e1_impl(inst, vm_state)?;
        }
        0x250 => {
            mul::execute_e1_impl(inst, vm_state)?;
        }
        0x251 | 0x252 | 0x253 => {
            mulh::execute_e1_impl(inst, vm_state)?;
        }
        0x254 | 0x255 | 0x256 | 0x257 => {
            divrem::execute_e1_impl(inst, vm_state)?;
        }
        0x260 | 0x261 => {
            hintstore::execute_e1_impl(inst, vm_state)?;
        }
        0 => {
            terminate_execute_e1_impl(inst, vm_state)?;
        }
        1 => {
            let discriminant = SysPhantom::from_repr(c as u16);
            if let Some(dis) = discriminant {
                match dis {
                    SysPhantom::Nop => nop_execute_e1_impl(inst, vm_state)?,
                    SysPhantom::DebugPanic => debug_panic_execute_e1_impl(inst, vm_state)?,
                    SysPhantom::CtStart | SysPhantom::CtEnd => nop_execute_e1_impl(inst, vm_state)?,
                }
            } else {
                openvm_circuit::system::phantom::execution::execute_e1_impl(inst, vm_state)?;
            }
        }
        _ => {
            println!("opcode: {opcode}");
            unreachable!()
        }
    }
    Ok(())
}
