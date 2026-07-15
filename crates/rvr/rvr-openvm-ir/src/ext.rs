/// Inline record shape emitted by a preflight extension instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineRecordShape {
    /// Two reads and one write, shared by ALU and LoadStore records.
    Alu3,
    /// Two reads and no write.
    Branch2,
    /// One conditional write.
    Wr1,
    /// One read and one conditional write.
    Rw1,
    /// Extension-owned record bytes. The extension emits the complete wire
    /// record and its registered assembler defines how those bytes are
    /// adopted by the consumer arena.
    Custom { record_size: usize },
}

/// Trait abstracting the code-generation context for extension instructions.
///
/// Extensions use this to read/write registers (with tracing handled in the
/// generated C code) and emit C lines, instead of writing raw C into a buffer.
/// Register access stays on the C side so the FFI boundary only carries
/// resolved values and memory.
pub trait ExtEmitCtx {
    /// Whether the current instruction is compiler-approved to emit its
    /// extension-owned inline record. Custom record emitters must consult
    /// this because whole-AIR tainting can fail a program slot back to logs.
    fn inline_record_enabled(&self) -> bool {
        false
    }

    /// Read a register with tracing. Returns a C expression for the value.
    fn read_reg(&mut self, idx: u8) -> String;

    /// Read a register while retaining the access in the ordinary memory
    /// chronology, and return `(value, from_timestamp, prev_timestamp)`.
    /// Custom direct-final records use the two timestamps in their record;
    /// non-preflight emitters return zero timestamp expressions.
    fn read_reg_with_trace(&mut self, idx: u8) -> (String, String, String);

    /// Read a register without tracing (for phantom instructions).
    fn read_reg_raw(&mut self, idx: u8) -> String;

    /// Write a register with tracing.
    fn write_reg(&mut self, idx: u8, val: &str);

    /// Write a register without tracing (for phantom instructions).
    fn write_reg_raw(&mut self, idx: u8, val: &str);

    /// Append a line of C code (indented).
    fn write_line(&mut self, s: &str);

    /// Read guest memory and return a C expression for the loaded value.
    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String;

    /// Write guest memory.
    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8);

    /// Emit an opaque C call that may update state observed by the tracer.
    fn extern_call(&mut self, name: &str, args: &[&str]);

    /// Emit a C call that does not require flushing local page trace state.
    fn extern_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        self.extern_call(name, args);
    }

    /// Emit an opaque C call that returns a value.
    fn extern_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String;

    /// Emit a chip-height update.
    fn trace_chip(&mut self, chip_idx: u32, count_expr: &str);

    /// Emit a single memory-page trace.
    fn trace_mem_access(&mut self, addr: &str, addr_space: u32);

    /// Emit a dword-range memory-page trace (rv64: each unit is 8 bytes).
    fn trace_mem_access_u64_range(&mut self, base_addr: &str, num_dwords: &str, addr_space: u32);

    /// Emit a value-carrying full-word (rv64: 8-byte, block-aligned) write
    /// trace to an arbitrary address space. Preflight logs a WRITE entry with
    /// the written value; metered records the page; pure is a no-op.
    fn trace_wr_as_u64(&mut self, addr: &str, val: &str, addr_space: u32);

    /// Trace a full-word store to an arbitrary address space and return the
    /// resolved `(src, ptr)` C expressions. Preflight codegen overrides this
    /// to emit an inline LoadStore record; other modes retain the verbose
    /// trace behavior below.
    fn trace_store_u64_as(
        &mut self,
        src_reg: u8,
        ptr_reg: u8,
        offset: u32,
        addr_space: u32,
        _local_opcode: u8,
    ) -> (String, String) {
        let ptr = self.read_reg(ptr_reg);
        let src = self.read_reg(src_reg);
        let addr = if offset == 0 {
            ptr.clone()
        } else {
            format!("({ptr} + 0x{offset:08x}u)")
        };
        self.trace_wr_as_u64(&addr, &src, addr_space);
        (src, ptr)
    }

    /// Emit a timestamp-only trace tick.
    fn trace_timestamp(&mut self);
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

    /// Compact record emitted by this extension in preflight mode, if any.
    fn inline_record_shape(&self) -> Option<InlineRecordShape> {
        None
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
