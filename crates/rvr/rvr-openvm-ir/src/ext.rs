use crate::MemWidth;

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

/// Trait abstracting the code-generation context for extension instructions.
///
/// Extensions use this context to read or write registers and emit C. The
/// emission mode decides whether register accesses are traced.
/// Register access stays on the C side so the FFI boundary only carries
/// resolved values and memory.
pub trait ExtEmitCtx {
    /// Read a register as an AIR-visible memory access.
    fn read_reg(&mut self, idx: u8) -> String;

    /// Get a register value without creating an AIR memory access. Value
    /// tracing records the value without advancing the memory timestamp.
    fn peek_reg(&mut self, idx: u8) -> String;

    /// Write a register, tracing it when required by the emission mode.
    fn write_reg(&mut self, idx: u8, val: &str);

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
    fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: u32);

    /// Record pages touched by a dword range for metering (one dword is 8 bytes).
    ///
    /// This records the address range, not the accessed values.
    fn trace_page_access_u64_range(&mut self, base_addr: &str, num_dwords: &str, addr_space: u32);
}

/// Trait for extension IR nodes. Implemented by each extension's instruction types.
pub trait ExtInstr: std::fmt::Debug + Send + Sync {
    /// Emit C code for this instruction via the emit context.
    ///
    /// Use `ctx.read_reg()` / `ctx.write_reg()` for register access (tracing
    /// is handled in the generated C code) and `ctx.write_line()` to emit raw
    /// C lines.
    fn emit_c(&self, ctx: &mut dyn ExtEmitCtx);

    /// Emit C code for a terminator extension instruction.
    ///
    /// `branch_to(target_pc)` returns a C tail-call statement (e.g.,
    /// `[[clang::musttail]] return block_0x...(state, ra, sp);`).
    /// Extensions that serve as block terminators (branches) should override this
    /// to emit comparison logic and use `branch_to` for control flow.
    ///
    /// Default: delegates to `emit_c`.
    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, _branch_to: &dyn Fn(u64) -> String) {
        self.emit_c(ctx);
    }

    /// Short op name for use in generated C comments (e.g. "keccakf").
    /// Default: returns `"ext"`.
    fn opname(&self) -> &str {
        "ext"
    }

    /// Whether this instruction may access main guest memory (`AS_MEMORY`).
    ///
    /// Codegen uses this to decide whether a metered block needs AS_MEMORY page
    /// tracking. The conservative default is `true` because most opaque
    /// extension FFIs read or write main memory through `state`.
    fn accesses_memory(&self) -> bool {
        true
    }

    /// Extra chip rows whose count is known when the artifact is generated.
    ///
    /// The generator adds them to the block's metering update, so the extension
    /// does not record them at runtime.
    fn fixed_trace_rows(&self) -> Vec<FixedTraceRows> {
        Vec::new()
    }

    /// Clone into a new boxed trait object.
    fn clone_box(&self) -> Box<dyn ExtInstr>;

    /// For terminator extensions: return successor PCs for CFG construction.
    /// Default returns `[fall_pc]` (fall-through behavior).
    fn successors(&self, fall_pc: u64) -> Vec<u64> {
        vec![fall_pc]
    }

    /// Whether this ends a basic block.
    /// Default is `false` (body instruction).
    fn is_block_end(&self) -> bool {
        false
    }
}

impl Clone for Box<dyn ExtInstr> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
