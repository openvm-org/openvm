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
/// recorded read or write advances it. A peek reads the current value and
/// preserves the current timestamp.
pub trait ExtEmitCtx {
    /// Read a variable through a VM memory access.
    fn read_var(&mut self, var: Variable) -> String;

    /// Read a variable at the current logical memory timestamp.
    fn peek_var(&mut self, var: Variable) -> String;

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

    /// Emit a dword-range memory trace (one dword is 8 bytes).
    fn trace_mem_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    );

    /// Emit a timestamp-only trace tick.
    fn trace_timestamp(&mut self);
}
