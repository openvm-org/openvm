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
    pre_compute_insns: Vec<PreComputeInstruction<F, Ctx>>,
    pc_start: u32,
}
```

Some notes, decisions made and the rationale
- `pre_compute_insns` is a `Vec` so that each thread has distinct pointers to their own `pre_compute_insns` 
- `pc_start` is not used at the moment and could be removed
- `lib` stores the dynamic library corresponding to this `AotInstance` which is already loaded during compile time

In the `AotInstance` the following methods are implemented:
- `new` creates a new instance for pure execution.
Important things to know about this:

This function generates x86 assembly source, then compiles it into a shared library via the `asm_to_lib` helper, which runs:
1. Writes the assembly to a temporary `.s` file
2. `gcc -fPIC -Wl,-z,noexecstack -shared <tmp>.s -o <tmp>.so`
3. Loads the resulting `.so` dynamic library

This function also calls `get_pre_compute_instructions` to generate the handler information for the fallback and stores it in a `Vec`.

Finally it returns an `AotInstance` which completes the compilation part for this given program.
- `execute_from_state`
Important things to know about this:

This function takes in `from_state: VmState<F, GuestMemory>` and `num_insns`. It will create a new `vm_exec_state` which is boxed and runs pure execution for `num_insns` instructions. Then it uses the information stored in `AotInstance`, specifically the `pre_compute_insns` and the dynamic library. We pass in the `VmExecState`, list of precompute instructions, initial pc and instret. We can potentially pass in more information by creating `Information` struct and passing in the pointer to that and then the assembly would "unpack" this information as it needs. Finally, we either return the `VmState` if the execution was successful or return some `ExecutionError` if it wasn't.
- `new_metered` creates a new instance for metered execution.
Important things to know about this:

This function follows the same build pipeline as `new` — it generates metered x86 assembly source, compiles it via `asm_to_lib` (`gcc -fPIC -shared`), and loads the resulting `.so` dynamic library.

There are also `set_pc` which will be called once at the end of the execution to sync the `VmExecState`'s pc from the x86 register. And also there is `should_suspend` which is called in every instruction and returns `1` if we should suspend and `0` otherwise which is later checked by the assembly.

Currently, the AOT feature is tested by executing on both interpreter and AOT in `air_test_impl` of `stark_utils.rs` and then asserting that the returned `instret`, `pc` `segments` and the register address space are equal.


