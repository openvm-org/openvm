use std::sync::Arc;

use openvm_instructions::exe::VmExe;
use openvm_platform::memory::MEM_SIZE;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_state::GuardedMemory;

use super::{
    build_callbacks, build_io_state, execute, execute_with_limit, register_and_execute,
    state::{init_rvr_state, state_as_void_ptr},
    PureTracer, PureTracerData, RvrCompiled,
};
use crate::{
    arch::{
        vm::{
            copy_guest_memory_to_rvr_memory, ensure_rvr_outcome, map_rvr_execute_error,
            read_public_values_from_guest_memory, read_rv32_regs_from_guest_memory, state_from_rvr,
            streams_from_io_state, streams_to_io_seed, write_rvr_memory_to_guest_memory,
        },
        ExecutionError, Streams, SystemConfig, VmState,
    },
    system::memory::online::GuestMemory,
};

pub struct RvrPureInstance<F> {
    pub(crate) system_config: SystemConfig,
    pub(crate) exe: Arc<VmExe<F>>,
    pub(crate) compiled: RvrCompiled,
}

impl<F> RvrPureInstance<F>
where
    F: PrimeField32,
{
    // TODO: deduplicate `execute` and `execute_from_state` — they share the rvr invocation
    // and result-translation logic and only differ in how the initial state is set up.
    pub fn execute(
        &self,
        inputs: impl Into<Streams<F>>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let inputs = inputs.into();
        let input_stream = inputs.input_stream;
        let hint_stream: Vec<u8> = inputs
            .hint_stream
            .into_iter()
            .map(|f| f.as_canonical_u32() as u8)
            .collect();

        if let Some(limit) = num_insns {
            let result = execute_with_limit(
                &self.compiled,
                self.exe.as_ref(),
                input_stream,
                hint_stream,
                limit,
                Default::default(),
            )
            .map_err(map_rvr_execute_error)?;
            return Ok(state_from_rvr(
                &self.system_config,
                self.exe.as_ref(),
                result.state.pc,
                &result.state.regs,
                &result.memory,
                &[],
            ));
        }

        let result = execute(
            &self.compiled,
            self.exe.as_ref(),
            input_stream,
            hint_stream,
            Default::default(),
        )
        .map_err(map_rvr_execute_error)?;

        Ok(state_from_rvr(
            &self.system_config,
            self.exe.as_ref(),
            result.state.pc,
            &result.state.regs,
            &result.memory,
            &result.public_values,
        ))
    }

    pub fn execute_from_state(
        &self,
        from_state: VmState<F, GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<VmState<F, GuestMemory>, ExecutionError> {
        let pc = from_state.pc();
        let mut guest_memory = from_state.memory;
        let (input_stream, hint_stream, deferrals) = streams_to_io_seed(from_state.streams);
        let rng = from_state.rng;
        #[cfg(feature = "metrics")]
        let metrics = from_state.metrics;

        let mut memory = GuardedMemory::new(MEM_SIZE)
            .map_err(|err| ExecutionError::RvrExecution(err.to_string()))?;

        let mut tracer_data = PureTracerData;
        let mut state = init_rvr_state(self.exe.as_ref(), &mut memory);
        state.tracer = PureTracer(&mut tracer_data);
        state.pc = pc;
        state
            .regs
            .copy_from_slice(&read_rv32_regs_from_guest_memory(&guest_memory));
        copy_guest_memory_to_rvr_memory(&guest_memory, &mut memory);
        match num_insns {
            Some(limit) => state.suspender.set_target(limit),
            None => state.suspender.disable(),
        }

        let mut io_state = build_io_state(input_stream, memory.as_mut_ptr(), Default::default());
        io_state.hint_stream = hint_stream;
        io_state.hint_pos = 0;
        io_state.public_values = read_public_values_from_guest_memory(&guest_memory);
        io_state.rng = rng;
        let callbacks = build_callbacks(&mut io_state);
        unsafe { register_and_execute(&self.compiled, &callbacks, state_as_void_ptr(&mut state)) }
            .map_err(map_rvr_execute_error)?;
        ensure_rvr_outcome(
            "execution from state",
            state.is_terminated(),
            state.is_suspended(),
            state.result_code(),
            num_insns.is_some(),
        )?;

        write_rvr_memory_to_guest_memory(
            &mut guest_memory,
            &state.regs,
            &memory,
            &io_state.public_values,
        );
        Ok(VmState::new(
            state.pc,
            guest_memory,
            streams_from_io_state(&io_state, deferrals),
            io_state.rng,
            #[cfg(feature = "metrics")]
            metrics,
        ))
    }
}
