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

/// Compact inline-record wire shapes. Each migrated instruction stores its
/// dynamic witness while the host re-derives program operands from PC.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InlineRecordShape {
    Alu3,
    Branch2,
    Wr1,
    Rw1,
    /// Extension-owned record bytes with an extension-defined consumer.
    Custom {
        record_size: usize,
    },
    /// Extension-owned packed records with a runtime row count. These records
    /// are valid only with a matching arena-native target: `capacity_per_row`
    /// sizes the backing from the metered AIR height, while the extension
    /// advances the target byte cursor by each record's actual packed size.
    CustomVariableRows {
        capacity_per_row: usize,
    },
}

/// Program-redundant fields required when an ALU compact record is emitted
/// directly into its arena-native record layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaAlu3Baked {
    pub rs2_field: u32,
    pub rs2_as: u8,
    pub rs2_imm_sign: u8,
    pub local_opcode: u8,
}

/// Program-redundant fields for the dedicated AddI arena-native record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaAddIBaked {
    pub imm_low11: u16,
    pub imm_sign: u16,
}

/// Program-redundant fields for a conditional one-write record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaWr1Baked {
    pub rd_ptr: u32,
    pub core_imm: u32,
    pub rd_data: u64,
    pub is_jal: u8,
    pub core_from_pc: u32,
}

/// Program-redundant fields for a read/conditional-write JALR record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArenaRw1Baked {
    pub rs1_ptr: u32,
    pub rd_ptr: u32,
    pub core_imm: u16,
    pub core_imm_sign: u8,
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
    /// Whether this instruction may emit its compiler-approved custom record.
    fn inline_record_enabled(&self) -> bool {
        false
    }

    /// Read a variable through a VM memory access.
    fn read_var(&mut self, var: Variable) -> String;

    /// Read a variable while retaining its ordinary memory chronology and
    /// return `(value, from_timestamp, previous_timestamp)`.
    fn read_var_with_trace(&mut self, var: Variable) -> (String, String, String) {
        (self.read_var(var), "0u".to_string(), "0u".to_string())
    }

    /// Read a variable at the current logical memory timestamp.
    fn peek_var(&mut self, var: Variable) -> String;

    /// Read a variable without emitting any trace event.
    fn read_var_raw(&mut self, var: Variable) -> String {
        self.peek_var(var)
    }

    /// Write a variable through a VM memory access.
    fn write_var(&mut self, var: Variable, val: &str);

    /// Write a variable without emitting any trace event.
    fn write_var_raw(&mut self, var: Variable, val: &str) {
        self.write_var(var, val);
    }

    /// Emit a compact two-read/one-write record using a result expression
    /// template. `__RVR_LHS__` and `__RVR_RHS__` are replaced with the traced
    /// operand expressions by the concrete emitter.
    fn emit_reg3_inline(
        &mut self,
        _rd: Variable,
        _rs1: Variable,
        _rs2: Variable,
        _arena: Option<ArenaAlu3Baked>,
        _result_template: &str,
    ) -> bool {
        false
    }

    /// Emit a compact register/immediate/write record. The result template
    /// uses the same placeholders as [`Self::emit_reg3_inline`].
    fn emit_reg2imm_inline(
        &mut self,
        _rd: Variable,
        _rs1: Variable,
        _imm_value: u64,
        _arena: Option<ArenaAlu3Baked>,
        _result_template: &str,
    ) -> bool {
        false
    }

    /// Emit the dedicated AddI compact record. Unlike the legacy mixed-ALU
    /// adapter, AddI consumes no timestamp for its immediate operand.
    fn emit_addi_inline(
        &mut self,
        _rd: Variable,
        _rs1: Variable,
        _imm_value: u64,
        _arena: Option<ArenaAddIBaked>,
    ) -> bool {
        false
    }

    /// Emit a compact conditional single-register write.
    fn emit_wr1_inline(
        &mut self,
        _rd: Option<Variable>,
        _value: &str,
        _arena: Option<ArenaWr1Baked>,
    ) -> bool {
        false
    }

    fn emit_load_inline(
        &mut self,
        _width: u8,
        _signed: bool,
        _rd: Option<Variable>,
        _base: Variable,
        _offset: i16,
    ) -> bool {
        false
    }

    fn emit_store_inline(
        &mut self,
        _width: u8,
        _base: Variable,
        _src: Variable,
        _offset: i16,
    ) -> bool {
        false
    }

    /// Append a line of C code (indented).
    fn write_line(&mut self, s: &str);

    /// Save execution-mode state and end the block through the shared RVR trap.
    fn emit_trap(&mut self);

    /// Read guest memory and return a C expression for the loaded value.
    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String;

    /// Write guest memory.
    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8);

    /// Reserve the optional second-block timestamp for a non-crossing RV64
    /// multi-byte access. The RV64 instruction node owns this adapter detail.
    fn trace_absent_second_block(&mut self, _base: &str, _offset: i16, _width: u8) {}

    /// Flush local page state, emit a C call, then reload the page state.
    fn emit_call(&mut self, name: &str, args: &[&str]);

    /// Emit an opaque C call that may update state observed by the tracer.
    fn extern_call(&mut self, name: &str, args: &[&str]) {
        self.emit_call(name, args);
    }

    /// Emit a C call that cannot access RVR state, without flushing page state.
    fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]);

    /// Emit an opaque C call without flushing local page trace state.
    fn extern_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        self.emit_call_without_page_flush(name, args);
    }

    /// Flush local page state, emit a C call that returns a value, then reload
    /// the page state.
    fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String;

    /// Emit an opaque C call that returns a value.
    fn extern_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
        self.emit_call_expr(ret_ty, name, args)
    }

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

    /// Emit a value-carrying full-word (rv64: 8-byte, block-aligned) write
    /// trace to an arbitrary address space. Preflight logs a WRITE entry with
    /// the written value; metered records the page; pure is a no-op.
    fn trace_wr_as_u64(&mut self, addr: &str, val: &str, addr_space: u32);

    /// Emit a full-word store and return the resolved `(src, ptr)` values.
    /// Preflight overrides this to emit the shared LoadStore compact record.
    fn trace_store_u64_as(
        &mut self,
        src: Variable,
        ptr: Variable,
        offset: i32,
        addr_space: u32,
        _local_opcode: u8,
    ) -> (String, String) {
        let ptr = self.read_var(ptr);
        let src = self.read_var(src);
        let addr = match offset.cmp(&0) {
            std::cmp::Ordering::Less => {
                format!("({ptr} - 0x{:08x}ull)", offset.unsigned_abs())
            }
            std::cmp::Ordering::Equal => ptr.clone(),
            std::cmp::Ordering::Greater => format!("({ptr} + 0x{offset:08x}ull)"),
        };
        self.trace_wr_as_u64(&addr, &src, addr_space);
        (src, ptr)
    }

    /// Emit a compact, implementation-owned REVEAL when the active transport
    /// can do so safely. Returning `true` means the complete instruction,
    /// including the external reveal callback, has been emitted.
    fn trace_reveal_compact(
        &mut self,
        _src: Variable,
        _ptr: Variable,
        _offset: i32,
        _width: u8,
        _addr_space: u32,
        _full_word_local_opcode: u8,
    ) -> bool {
        false
    }

    /// Emit a timestamp-only trace tick.
    fn trace_timestamp(&mut self);

    /// Emit the fixed Phantom consumer record for the current instruction and
    /// advance its single timestamp. Preflight codegen overrides this to use
    /// the compiler-selected whole-AIR direct-final target; other contexts
    /// retain the ordinary timestamp-only behavior.
    fn trace_phantom_record(&mut self, _operands: [u32; 3]) {
        self.trace_timestamp();
    }
}
