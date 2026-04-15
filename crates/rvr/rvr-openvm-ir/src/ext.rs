/// Trait abstracting the code-generation context for extension instructions.
///
/// Extensions use this to read/write registers (with tracing handled in the
/// generated C code) and emit C lines, instead of writing raw C into a buffer.
/// Register access stays on the C side so the FFI boundary only carries
/// resolved values and memory.
pub trait ExtEmitCtx {
    /// Read a register with tracing. Returns a C expression for the value.
    fn read_reg(&mut self, idx: u8) -> String;
    /// Read a register without tracing (for phantom instructions).
    fn read_reg_raw(&mut self, idx: u8) -> String;
    /// Write a register with tracing.
    fn write_reg(&mut self, idx: u8, val: &str);
    /// Write a register without tracing (for phantom instructions).
    fn write_reg_raw(&mut self, idx: u8, val: &str);
    /// Append a line of C code (indented).
    fn write_line(&mut self, s: &str);
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
    fn emit_c_term(&self, ctx: &mut dyn ExtEmitCtx, _branch_to: &dyn Fn(u32) -> String) {
        self.emit_c(ctx);
    }

    /// Short op name for use in generated C comments (e.g. "keccakf").
    /// Default: returns `"ext"`.
    fn opname(&self) -> &str {
        "ext"
    }

    /// Clone into a new boxed trait object.
    fn clone_box(&self) -> Box<dyn ExtInstr>;

    /// For terminator extensions: return successor PCs for CFG construction.
    /// Default returns `[fall_pc]` (fall-through behavior).
    fn successors(&self, fall_pc: u32) -> Vec<u32> {
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
