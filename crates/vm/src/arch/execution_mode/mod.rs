pub mod e1;
pub mod metered;
pub mod tracegen;

// TODO(ayush): better name
pub trait E1E2ExecutionCtx {
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32);
}
