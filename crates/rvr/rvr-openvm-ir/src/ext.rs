use crate::{MemWidth, ValueSlot};

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
/// Main memory is distinguished because generated code keeps its current page
/// in local state and must flush that state around calls that can change it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PageAddressSpace {
    /// The target's main guest-memory address space.
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

/// Trait abstracting the code-generation context for extension instructions.
///
/// Extensions use this context to access opaque value slots and emit C. The
/// emission mode decides whether accesses are traced.
pub trait ExtEmitCtx {
    /// Read a value slot as an AIR-visible memory access.
    fn read_slot(&mut self, slot: ValueSlot) -> String;
    /// Get a slot value without creating an AIR memory access.
    /// Value tracing records the value without advancing the memory timestamp.
    fn peek_slot(&mut self, slot: ValueSlot) -> String;
    /// Write a value slot, tracing it when required by the emission mode.
    fn write_slot(&mut self, slot: ValueSlot, val: &str);
    /// Append a line of C code.
    fn write_line(&mut self, s: &str);
    /// Save execution-mode state and end the block through the shared RVR trap.
    fn emit_trap(&mut self);
    /// Read guest memory and return a C expression for the loaded value.
    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String;
    /// Write guest memory.
    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8);
    /// Flush local page state, emit a C call, then reload the page state.
    fn emit_call(&mut self, name: &str, args: &[&str]);
    /// Emit a C call that cannot access RVR state, without flushing page state.
    fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]);
    /// Flush local page state, emit a C call that returns a value, then reload
    /// the page state.
    fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String;
    /// Emit a call and save its result only when chip tracing needs it.
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
    /// This records the address, not the accessed value.
    fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace);
    /// Record pages touched by a dword range for metering (one dword is 8 bytes).
    /// This records the address range, not the accessed values.
    fn trace_page_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    );
}
