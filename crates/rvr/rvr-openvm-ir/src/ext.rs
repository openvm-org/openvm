use crate::{MemWidth, Variable};

/// Extra trace rows added by one extension instruction.
///
/// The PC-to-chip mapping already counts the instruction's main row. This
/// records any other rows whose count is known when the C code is generated.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedTraceRows {
    /// AIR index whose trace height increases.
    pub chip_idx: u32,
    /// Number of additional rows contributed by one instruction.
    pub count: u32,
}

/// Address space classification used by page-access metering.
///
/// Main memory is distinct because generated metered code caches its current
/// page locally. Calls that may access main memory require this cache to be
/// flushed before the call and reloaded afterward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageAddressSpace {
    /// The main guest-memory address space.
    MainMemory(u32),
    /// Any other address space identified by its numeric ID.
    Other(u32),
}

impl PageAddressSpace {
    pub const fn id(self) -> u32 {
        match self {
            Self::MainMemory(id) | Self::Other(id) => id,
        }
    }

    pub const fn is_main_memory(self) -> bool {
        matches!(self, Self::MainMemory(_))
    }
}

/// Code-generation context used by instruction nodes.
///
/// Value tracing emits ordered hooks used to build execution records. The
/// logical memory timestamp represents the order of VM memory accesses. Each
/// recorded read or write advances it, and instructions may reserve disabled
/// slots without a memory event. A peek reads the current value and preserves
/// the current timestamp.
pub trait ExtEmitCtx {
    /// Whether this emitter is producing the minimal preflight transcript.
    fn is_preflight(&self) -> bool {
        false
    }

    /// Read a variable through a VM memory access.
    fn read_var(&mut self, var: Variable) -> String;

    /// Read a variable at the current logical memory timestamp.
    fn peek_var(&mut self, var: Variable) -> String;

    /// Account for logical memory slots not emitted through `read_var`,
    /// `write_var`, `read_mem`, or `write_mem`.
    ///
    /// Pure and metered emitters preserve their existing execution behavior.
    fn advance_timestamp(&mut self, slots: u32);

    /// Write a variable through a VM memory access.
    fn write_var(&mut self, var: Variable, val: &str);

    /// Append a line of C code (indented).
    fn write_line(&mut self, s: &str);

    /// Save execution-mode state and end the block through the shared RVR trap.
    fn emit_trap(&mut self);

    /// Read guest memory and return a C expression for the loaded value.
    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String;

    /// Write guest memory.
    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8);

    /// Write one naturally aligned eight-byte main-memory block.
    ///
    /// Unlike a scalar doubleword store, this is one enabled memory event and
    /// does not reserve a second block-access slot.
    fn write_aligned_mem_block(&mut self, addr: &str, val: &str);

    /// Ensure preflight can advance `slots` logical clock slots before an
    /// instruction starts mutating state.
    ///
    /// Pure and metered emitters preserve their existing execution behavior.
    fn reserve_preflight_timestamp_slots(&mut self, slots: &str);

    /// Reserve space for a runtime-sized sequence of replay values.
    ///
    /// Preflight uses this before consuming host advice. Metered
    /// execution uses the same count without materializing a replay-value stream.
    fn reserve_replay_values(&mut self, _count: &str) {}

    /// Count a statically known number of replay values without materializing
    /// them. Metered execution folds these counts into one update per block.
    fn count_fixed_replay_values(&mut self, _count: u32) {}

    /// Count a runtime-sized replay-value sequence after its producing
    /// operation succeeds, without opening a materialization reservation.
    fn count_replay_values(&mut self, _count: &str) {}

    /// Append one architectural `u64` value to the replay-value stream.
    ///
    /// Values are untagged and ordered by execution. Checkpoint replay knows
    /// which instruction consumes each value from the program itself.
    fn append_replay_value(&mut self, _value: &str) {}

    /// Append a post-write range of aligned main-memory words to checkpoint
    /// replay values.
    ///
    /// This emits no loads outside preflight; metered execution only
    /// accounts for the already-reserved range length.
    fn append_replay_memory_u64_range(&mut self, _base: &str, _count: &str) {}

    /// Commit mode-local execution metadata before emitting a control transfer.
    ///
    /// Instruction-owned terminators must call this after their final logged
    /// access or replay value and before writing any branch or return.
    fn flush_before_control_transfer(&mut self) {}

    /// Flush local page state, emit a C call, then reload the page state.
    fn emit_call(&mut self, name: &str, args: &[&str]);

    /// Emit a C call that cannot access RVR state, without flushing page state.
    fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]);

    /// Flush local page state, emit a C call that returns a value, then reload
    /// the page state.
    fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String;

    /// Emit a call and save its result only when chip tracing needs it.
    ///
    /// Pure execution emits the call as a statement and returns `None`.
    fn emit_call_with_trace_result(
        &mut self,
        ret_ty: &str,
        name: &str,
        args: &[&str],
    ) -> Option<String>;

    /// Emit a call that can access RVR state and trap if it returns `false`.
    fn emit_checked_call(&mut self, name: &str, args: &[&str]) {
        let result = self.emit_call_expr("bool", name, args);
        self.write_line(&format!("if (unlikely(!{result})) {{"));
        self.emit_trap();
        self.write_line("}");
    }

    /// Emit a call without flushing page state and trap if it returns `false`.
    fn emit_checked_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        self.write_line(&format!("if (unlikely(!{name}({}))) {{", args.join(", ")));
        self.emit_trap();
        self.write_line("}");
    }

    /// Emit a chip-height update.
    fn trace_chip(&mut self, chip_idx: u32, count_expr: &str);

    /// Emit a chip-height update only when `count_expr` is nonzero.
    fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str);

    /// Record the pages containing one fixed-width access for metering.
    ///
    /// This records the address, not the accessed value.
    fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace);

    /// Record pages touched by a dword range for metering (one dword is 8 bytes).
    ///
    /// This records the address range, not the accessed values.
    fn trace_page_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    );
}
