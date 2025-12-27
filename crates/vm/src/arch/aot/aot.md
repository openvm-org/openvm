### Documentation of the current AOT pipeline

There is an `AotInstance` struct which stores the information generated during compile time to be used in execution time.
```
pub struct AotInstance<F, Ctx> {
    init_memory: SparseMemoryImage,
    system_config: SystemConfig,
    // SAFETY: this is not actually dead code, but `pre_compute_insns` contains raw pointer refers
    // to this buffer.
    #[allow(dead_code)]
    pre_compute_buf: AlignedBuf,
    lib: Library,
    pre_compute_insns_box: Box<[PreComputeInstruction<F, Ctx>]>,
    pc_start: u32,
}
```

Some notes, decisions made and the rationale
- `pre_compute_insns` is boxed so that each thread have distinct pointers to their own `pre_compute_insns` 
- `pc_start` is not used at the moment and could be removed
- `lib` stores the dynamic library corresponding to this `AotInstance` which is already loaded during compile time

In the `AotInstance` the following methods are implemented:
- `new` creates a new instance for pure execution, where the asm will be created with the default name `asm_x86_run` 
- `new_with_asm_name` 
Important things to know about this:

This function will run shell commands
1. `as src/{asm_name}.s -o {asm_name}.o` this command will be ran from `crates/vm/src/arch/asm_bridge` directory
2. `ar rcs lib{asm_name}.a {asm_name}.o` this command will be ran from `crates/vm/src/arch/asm_bridge` directory
3. `cargo rustc --release --target-dir={root_dir}/target/{asm_name} -L crates/vm/src/arch/asm_bridge -l static={asm_name}` this command will be ran from `crates/vm/src/arch/asm_bridge` directory. `{root_dir}` is the location of the root `openvm` crate. this command finds the `lib{asm_name}.a` static link at the `crates/vm/src/arch/asm_bridge` directory and then places the resulting `libasm_bridge.so` dynamic library at the target dir which is `{root_dir}/target/{asm_name}/release/libasm_bridge.so`

This function also calls `get_pre_compute_instructions` to generate the handler information for the fallback and stores it in a box.

Finally it returns an `AotInstance` which completes the compilation part of for this given program.
- `execute_from_state`
Important things to know about this:

This function takes in `from_state: VmState<F, GuestMemory>` and `num_insns`. It will create a new `vm_exec_state` which is boxed and runs pure execution for `num_insns` instructions. Then it uses the information stored in `AotInstance`, specifically the `pre_compute_insns` and the dynamic library. We pass in the `VmExecState`, list of precompute instructions, initial pc and instret. We can potentially pass in more information by creating `Information` struct and passing in the pointer to that and then the assembly would "unpack" this information as it needs. Finally, we either return the `VmState` if the execution was successful or return some `ExecutionError` if it wasn't.
- `new_metered` creates a new instance for metered execution, where the asm will be created with the default name `asm_x86_run` 
- `new_metered_with_asm_name` 
Important things to knouw about this:

This function will run shell commands
1. `as src/{asm_name}.s -o {asm_name}.o` this command will be ran from `crates/vm/src/arch/asm_bridge_metered` directory 
2. `ar rcs lib{asm_name}.a {asm_name}.o` this command will be ran from `crates/vm/src/arch/asm_bridge_metered` directory 
3. `cargo rustc --release --target-dir={root_dir}/target/{asm_name} -L crates/vm/src/arch/asm_bridge_metered -l static={asm_name}` this command will be ran from `crates/vm/src/arch/asm_bridge_metered` directory.  `{root_dir}` is the location of the root `openvm` crate. this command finds the `lib{asm_name}.a` static link at the `crates/vm/src/arch/asm_bridge_metered` directory and then places the resulting `libasm_bridge_metered.so` dynamic library at the target dir which is `{root_dir}/target/{asm_name}/release/libasm_bridge_metered.so`

The next important thing to know is the `asm_bridge` and `asm_bridge_metered` crate. We build these crates to obtain the dynamic library. Inside this library contains the assembly and `lib.rs` file. Currently the assembly is constant for any program since this is the simplest form of the interpreter fallback. But for the next steps, the assembly won't be constant since each instruction would get its own pc flag and for the RV32 instructions, they will not use the extern fallback and directly perform the operation in the assembly. 

In the `lib.rs` files contain the rust functions that would call the rust implementations of these functions. Note that `asm_bridge` and `asm_bridge_metered` is a workspace member of the `openvm` crate which allows us to just import the circuit functions and struct and dereference the `vm_exec_state` and `pre_compute_insns` we stored. 

There are also `set_pc` which will be called once at the end of the execution to sync the `VmExecState`'s pc from the x86 register. And also there is `should_suspend` which is called in every instruction and returns `1` if we should suspend and `0` otherwise which is later checked by the assembly.

Currently, the AOT feature is tested by executing on both interpreter and AOT in `air_test_impl` of `stark_utils.py` and then asserting that the returned `instret`, `pc` `segments` and the register address space are equal.


