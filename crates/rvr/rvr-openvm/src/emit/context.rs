use std::{collections::HashSet, fmt::Write};

use super::codegen::hex_u32;

/// Code generation context. Holds a mutable buffer and tracks hot registers.
pub struct EmitContext {
    buf: String,
    hot_regs: HashSet<u8>,
    /// Base indentation level (in 4-space units) for emitted lines.
    indent: usize,
    /// Counter for unique variable names.
    var_counter: u32,
}

impl EmitContext {
    pub fn new(hot_regs: HashSet<u8>) -> Self {
        Self {
            buf: String::with_capacity(1024),
            hot_regs,
            indent: 2,
            var_counter: 0,
        }
    }

    fn next_var(&mut self) -> String {
        let id = self.var_counter;
        self.var_counter += 1;
        format!("_v{id}")
    }

    pub fn materialize_u32(&mut self, expr: &str) -> String {
        let tmp = self.next_var();
        self.write_line(&format!("uint32_t {tmp} = {expr};"));
        tmp
    }

    pub fn take_buf(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    pub fn buf(&self) -> &str {
        &self.buf
    }

    pub fn buf_mut(&mut self) -> &mut String {
        &mut self.buf
    }

    /// Append a line of C code (indented).
    pub fn write_line(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(s);
        self.buf.push('\n');
    }

    /// Public ABI name accessor for use by project.rs.
    pub fn abi_name_static(reg: u8) -> &'static str {
        Self::abi_name(reg)
    }

    fn abi_name(reg: u8) -> &'static str {
        match reg {
            0 => "zero",
            1 => "ra",
            2 => "sp",
            3 => "gp",
            4 => "tp",
            5 => "t0",
            6 => "t1",
            7 => "t2",
            8 => "s0",
            9 => "s1",
            10 => "a0",
            11 => "a1",
            12 => "a2",
            13 => "a3",
            14 => "a4",
            15 => "a5",
            16 => "a6",
            17 => "a7",
            18 => "s2",
            19 => "s3",
            20 => "s4",
            21 => "s5",
            22 => "s6",
            23 => "s7",
            24 => "s8",
            25 => "s9",
            26 => "s10",
            27 => "s11",
            28 => "t3",
            29 => "t4",
            30 => "t5",
            31 => "t6",
            _ => unreachable!(),
        }
    }

    /// Read a register with tracing. Returns a C expression for the value.
    pub fn read_reg(&mut self, idx: u8) -> String {
        if idx == 0 {
            return "0".to_string();
        }
        if self.hot_regs.contains(&idx) {
            let name = Self::abi_name(idx);
            self.write_line(&format!("trace_reg_read(state, {idx}, {name});"));
            name.to_string()
        } else {
            let var = self.next_var();
            self.write_line(&format!("uint32_t {var} = reg_read(state, {idx});"));
            self.write_line(&format!("trace_reg_read(state, {idx}, {var});"));
            var
        }
    }

    /// Read a register WITHOUT tracing (for phantom instructions).
    pub fn read_reg_raw(&mut self, idx: u8) -> String {
        if idx == 0 {
            return "0".to_string();
        }
        if self.hot_regs.contains(&idx) {
            Self::abi_name(idx).to_string()
        } else {
            format!("state->regs[{idx}]")
        }
    }

    /// Write a register with tracing. Trace before write so tracer can read
    /// the old value from state if needed.
    ///
    /// `val` is materialized into a temporary first so it is evaluated exactly
    /// once even when interpolated into both the trace and the data write
    /// statements. This matters when `val` is an opaque function call (e.g.
    /// an `rvr_ext_*` extension entry point) that the C compiler cannot CSE.
    pub fn write_reg(&mut self, idx: u8, val: &str) {
        if idx == 0 {
            return;
        }
        let tmp = self.next_var();
        self.write_line(&format!("uint32_t {tmp} = {val};"));
        self.write_line(&format!("trace_reg_write(state, {idx}, {tmp});"));
        if self.hot_regs.contains(&idx) {
            let name = Self::abi_name(idx);
            self.write_line(&format!("{name} = {tmp};"));
        } else {
            self.write_line(&format!("reg_write(state, {idx}, {tmp});"));
        }
    }

    /// Write a register WITHOUT tracing (for phantom instructions).
    pub fn write_reg_raw(&mut self, idx: u8, val: &str) {
        if idx == 0 {
            return;
        }
        if self.hot_regs.contains(&idx) {
            let name = Self::abi_name(idx);
            self.write_line(&format!("{name} = {val};"));
        } else {
            self.write_line(&format!("state->regs[{idx}] = {val};"));
        }
    }

    fn addr_expr(base: &str, offset: i16) -> String {
        if offset == 0 {
            base.to_string()
        } else if offset > 0 {
            format!("{base} + {}", hex_u32(offset as u32))
        } else {
            format!("{base} - {}", hex_u32((-(offset as i32)) as u32))
        }
    }

    /// Read memory with tracing (per-width specialization). Returns a C expression.
    pub fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        let addr = Self::addr_expr(base, offset);
        let var = self.next_var();
        let (data_func, var_ty) = match (width, signed) {
            (1, false) => ("rd_mem_u8_traced", "uint32_t"),
            (1, true) => ("rd_mem_i8_traced", "int32_t"),
            (2, false) => ("rd_mem_u16_traced", "uint32_t"),
            (2, true) => ("rd_mem_i16_traced", "int32_t"),
            (4, _) => ("rd_mem_u32_traced", "uint32_t"),
            _ => unreachable!("invalid memory width {width}"),
        };
        self.write_line(&format!("{var_ty} {var} = {data_func}(state, {addr});"));
        var
    }

    /// Write memory. The combined helper traces before storing; passing
    /// `val` as an argument evaluates it exactly once, so no local temp
    /// is needed even when `val` is an opaque expression.
    pub fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        let addr = Self::addr_expr(base, offset);
        let (cast_ty, wr_func) = match width {
            1 => ("uint8_t", "wr_mem_u8_traced"),
            2 => ("uint16_t", "wr_mem_u16_traced"),
            4 => ("uint32_t", "wr_mem_u32_traced"),
            _ => unreachable!("invalid memory width {width}"),
        };
        self.write_line(&format!("{wr_func}(state, {addr}, ({cast_ty})({val}));"));
    }

    /// Emit a trace_pc call. Per-instruction chip accounting is rolled into
    /// the per-block chip update emitted at block entry by
    /// `CProject::emit_block_function`, not here.
    pub fn trace_pc(&mut self, pc: u32) {
        self.write_line(&format!("trace_pc(state, 0x{pc:08x}u);"));
    }

    pub fn extern_call(&mut self, name: &str, args: &[&str]) {
        let args_str = args.join(", ");
        self.write_line(&format!("{name}({args_str});"));
    }

    pub fn trace_mem_access(&mut self, addr: &str, addr_space: u32) {
        self.write_line(&format!("trace_mem_access(state, {addr}, {addr_space}u);"));
    }

    fn sorted_hot_regs(&self) -> Vec<u8> {
        let mut regs: Vec<u8> = self.hot_regs.iter().copied().collect();
        regs.sort();
        regs
    }

    pub fn tail_call_args(&self) -> String {
        let mut args = "state".to_string();
        for &idx in &self.sorted_hot_regs() {
            let name = Self::abi_name(idx);
            write!(args, ", {name}").unwrap();
        }
        args
    }

    pub fn sync_regs_to_state(&mut self) {
        let args = self.tail_call_args();
        self.write_line(&format!("rv_save_hot_regs({args});"));
    }

    pub fn sync_regs_from_state(&mut self) {
        for &idx in &self.sorted_hot_regs() {
            let name = Self::abi_name(idx);
            self.write_line(&format!("{name} = state->regs[{idx}];"));
        }
    }
}

impl rvr_openvm_ir::ExtEmitCtx for EmitContext {
    fn read_reg(&mut self, idx: u8) -> String {
        EmitContext::read_reg(self, idx)
    }

    fn read_reg_raw(&mut self, idx: u8) -> String {
        EmitContext::read_reg_raw(self, idx)
    }

    fn write_reg(&mut self, idx: u8, val: &str) {
        EmitContext::write_reg(self, idx, val)
    }

    fn write_reg_raw(&mut self, idx: u8, val: &str) {
        EmitContext::write_reg_raw(self, idx, val)
    }

    fn write_line(&mut self, s: &str) {
        EmitContext::write_line(self, s)
    }
}
