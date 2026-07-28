use std::{marker::PhantomData, sync::Arc};

use openvm_instructions::exe::VmExe;
use openvm_stark_backend::p3_field::PrimeField32;

use super::{
    execution_mode::{PreflightCtx, Segment},
    interpreter::InterpretedInstance,
    ExecutionError, Executor, ExecutorInventory, PreflightOutput, StaticProgramError, Streams,
    VmState,
};
use crate::system::memory::online::GuestMemory;

/// Owned interpreter instance for append-only preflight execution.
///
/// The ordinary interpreter borrows its executor inventory because generated
/// pre-compute data may point into executors. Proving instances own both the VM
/// and this interpreter, so this wrapper keeps the shared inventory alive for
/// exactly as long as the borrowed interpreter.
pub struct PreflightInterpretedInstance<F, E> {
    // Drop the interpreter before releasing the inventory that backs its
    // pre-compute pointers.
    inner: InterpretedInstance<'static, PreflightCtx>,
    _inventory: Arc<ExecutorInventory<E>>,
    _field: PhantomData<fn() -> F>,
}

impl<F, E> PreflightInterpretedInstance<F, E>
where
    F: PrimeField32,
    E: Executor<F> + 'static,
{
    pub fn new(
        exe: &VmExe<F>,
        inventory: Arc<ExecutorInventory<E>>,
    ) -> Result<Self, StaticProgramError> {
        let inventory_ref = unsafe {
            // SAFETY:
            // - `inventory` is stored in the returned wrapper and therefore outlives `inner`.
            // - `inner` is declared before `_inventory`, so it is dropped first.
            // - moving the Arc does not move its allocation.
            &*Arc::as_ptr(&inventory)
        };
        let inner = InterpretedInstance::new(inventory_ref, exe)?;
        Ok(Self {
            inner,
            _inventory: inventory,
            _field: PhantomData,
        })
    }

    pub fn create_initial_vm_state(&self, inputs: impl Into<Streams>) -> VmState<GuestMemory> {
        self.inner.create_initial_vm_state(inputs)
    }

    /// Executes exactly one metered segment from its architectural start state.
    pub fn execute_segment(
        &self,
        state: VmState<GuestMemory>,
        segment: &Segment,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.inner.execute_segment::<F>(state, segment)
    }

    /// Low-level interpreter entry point.
    ///
    /// `None` runs until termination and may retain an unbounded history. Normal
    /// proving code should use [`Self::execute_segment`] with a metered bound.
    pub fn execute_preflight_from_state(
        &self,
        state: VmState<GuestMemory>,
        num_insns: Option<u64>,
    ) -> Result<PreflightOutput, ExecutionError> {
        self.inner
            .execute_preflight_from_state::<F>(state, num_insns)
    }
}
