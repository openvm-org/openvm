use std::{collections::HashSet, fmt::Write};

use rvr_openvm_ir::{MemWidth, PageAddressSpace, Variable};

use super::codegen::hex_u32;

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
#[error("chip index {chip_idx} is outside AIR count {num_airs}")]
pub(crate) struct InvalidChipIndex {
    pub chip_idx: u32,
    pub num_airs: u32,
}

pub(crate) fn validate_chip_index(chip_idx: u32, num_airs: u32) -> Result<(), InvalidChipIndex> {
    if chip_idx < num_airs {
        Ok(())
    } else {
        Err(InvalidChipIndex { chip_idx, num_airs })
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum EmitMode {
    /// Emit ordered register, PC, and memory-value hooks.
    ///
    /// Page hooks are separate because they record only addresses for metering.
    #[allow(dead_code)]
    ValueTrace,
    /// Memory accesses use direct helpers and do not emit memory trace events.
    #[default]
    Direct,
    /// Metered block ABI. Blocks with memory ops record AS_MEMORY pages locally.
    Metered { trace_memory_pages: bool },
    /// Metered-cost execution with chip widths written into generated C.
    MeteredCost,
}

/// Extra values carried between generated blocks through the `preserve_none`
/// tail-call ABI.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum BlockAbi {
    /// State pointer and hot guest registers only.
    #[default]
    Plain,
    /// Pure execution with an instret countdown.
    InstretCountdown,
    /// Metered execution with its periodic-check counter and trace heights.
    Metered,
}

impl BlockAbi {
    pub(crate) const fn extra_args(self) -> usize {
        match self {
            Self::Plain => 0,
            Self::InstretCountdown => 1,
            Self::Metered => 2,
        }
    }
}

impl EmitMode {
    fn traces_values(self) -> bool {
        matches!(self, Self::ValueTrace)
    }

    fn is_metered_without_memory_pages(self) -> bool {
        matches!(
            self,
            Self::Metered {
                trace_memory_pages: false
            }
        )
    }

    /// Whether this block records AS_MEMORY pages through local `TraceMemory`.
    fn traces_memory_pages(self) -> bool {
        matches!(
            self,
            Self::Metered {
                trace_memory_pages: true
            }
        )
    }
}

#[derive(Clone, Copy)]
enum RegisterReadKind {
    MemoryAccess,
    Peek,
}

/// Code generation context. Holds a mutable buffer and tracks hot registers.
pub struct EmitContext<'a> {
    buf: String,
    hot_regs: HashSet<u8>,
    /// Base indentation level (in 4-space units) for emitted lines.
    indent: usize,
    /// Counter for unique variable names.
    var_counter: u32,
    uses_raw_memory: bool,
    mode: EmitMode,
    block_abi: BlockAbi,
    chip_widths: Option<&'a [u64]>,
    num_airs: Option<u32>,
    invalid_chip_index: Option<InvalidChipIndex>,
}

impl<'a> EmitContext<'a> {
    pub(crate) fn new(
        hot_regs: HashSet<u8>,
        mode: EmitMode,
        block_abi: BlockAbi,
        chip_widths: Option<&'a [u64]>,
        num_airs: Option<u32>,
    ) -> Self {
        debug_assert_eq!(matches!(mode, EmitMode::MeteredCost), chip_widths.is_some());
        Self {
            buf: String::with_capacity(1024),
            hot_regs,
            indent: 2,
            var_counter: 0,
            uses_raw_memory: false,
            mode,
            block_abi,
            chip_widths,
            num_airs,
            invalid_chip_index: None,
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

    pub fn materialize_u64(&mut self, expr: &str) -> String {
        let tmp = self.next_var();
        self.write_line(&format!("uint64_t {tmp} = {expr};"));
        tmp
    }

    pub fn take_buf(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }

    pub(crate) fn uses_raw_memory(&self) -> bool {
        self.uses_raw_memory
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

    /// Save local metering state and tail-call the shared RVR trap.
    pub fn emit_trap(&mut self) {
        self.flush_page_locals();
        let args = self.tail_call_args();
        self.write_line(&format!("[[clang::musttail]] return rv_trap({args});"));
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

    fn read_reg_impl(&mut self, idx: u8, kind: RegisterReadKind) -> String {
        if idx == 0 {
            return "0ull".to_string();
        }

        let value = if self.hot_regs.contains(&idx) {
            let name = Self::abi_name(idx);
            name.to_string()
        } else {
            let var = self.next_var();
            self.write_line(&format!("uint64_t {var} = reg_read(state, {idx});"));
            var
        };

        if self.mode.traces_values() {
            let trace_fn = match kind {
                RegisterReadKind::MemoryAccess => "trace_reg_read",
                RegisterReadKind::Peek => "trace_reg_peek",
            };
            self.write_line(&format!("{trace_fn}(state, {idx}, {value});"));
        }

        value
    }

    /// Read an AIR-visible register value.
    pub fn read_reg(&mut self, idx: u8) -> String {
        self.read_reg_impl(idx, RegisterReadKind::MemoryAccess)
    }

    /// Get a register value without creating a VM memory access.
    pub fn peek_reg(&mut self, idx: u8) -> String {
        self.read_reg_impl(idx, RegisterReadKind::Peek)
    }

    /// Write a register with tracing when value tracing is enabled.
    pub fn write_reg(&mut self, idx: u8, val: &str) {
        if idx == 0 {
            return;
        }
        if self.mode.traces_values() {
            let tmp = self.next_var();
            self.write_line(&format!("uint64_t {tmp} = (uint64_t)({val});"));
            self.write_line(&format!("trace_reg_write(state, {idx}, {tmp});"));
            if self.hot_regs.contains(&idx) {
                let name = Self::abi_name(idx);
                self.write_line(&format!("{name} = {tmp};"));
            } else {
                self.write_line(&format!("reg_write(state, {idx}, {tmp});"));
            }
            return;
        }
        self.write_reg_direct(idx, &format!("(uint64_t)({val})"));
    }

    fn write_reg_direct(&mut self, idx: u8, val: &str) {
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

    fn read_mem_helper(width: u8, signed: bool) -> (&'static str, &'static str, &'static str) {
        match (width, signed) {
            (1, false) => ("read_mem_u8", "trace_read_mem_u8", "uint32_t"),
            (1, true) => ("read_mem_i8", "trace_read_mem_i8", "int32_t"),
            (2, false) => ("read_mem_u16", "trace_read_mem_u16", "uint32_t"),
            (2, true) => ("read_mem_i16", "trace_read_mem_i16", "int32_t"),
            (4, false) => ("read_mem_u32", "trace_read_mem_u32", "uint32_t"),
            (4, true) => ("read_mem_i32", "trace_read_mem_i32", "int32_t"),
            (8, _) => ("read_mem_u64", "trace_read_mem_u64", "uint64_t"),
            _ => unreachable!("invalid memory width {width}"),
        }
    }

    fn write_mem_helper(width: u8) -> (&'static str, &'static str, &'static str) {
        match width {
            1 => ("write_mem_u8", "trace_write_mem_u8", "uint8_t"),
            2 => ("write_mem_u16", "trace_write_mem_u16", "uint16_t"),
            4 => ("write_mem_u32", "trace_write_mem_u32", "uint32_t"),
            8 => ("write_mem_u64", "trace_write_mem_u64", "uint64_t"),
            _ => unreachable!("invalid memory width {width}"),
        }
    }

    /// Read guest memory. Metered hot blocks record the memory page separately.
    pub fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        assert!(
            !self.mode.is_metered_without_memory_pages(),
            "metered memory read emitted without page tracking"
        );
        let addr = Self::addr_expr(base, offset);
        let var = self.next_var();
        let (read_func, trace_func, var_ty) = Self::read_mem_helper(width, signed);
        self.uses_raw_memory = true;

        self.write_line(&format!("{var_ty} {var} = {read_func}(memory, {addr});"));
        if self.mode.traces_values() {
            self.write_line(&format!("{trace_func}(state, {addr}, {var});"));
        }
        if self.mode.traces_memory_pages() {
            self.emit_inline_page_record(&addr, width);
        }
        var
    }

    /// Emit a guest memory write. Metered hot blocks record the memory page
    /// through the block-local `TraceMemory` context, then use the raw memory
    /// helper so the common path avoids tracing calls.
    pub fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        assert!(
            !self.mode.is_metered_without_memory_pages(),
            "metered memory write emitted without page tracking"
        );
        let addr = Self::addr_expr(base, offset);
        let (write_func, trace_func, cast_ty) = Self::write_mem_helper(width);
        self.uses_raw_memory = true;

        if self.mode.traces_memory_pages() {
            self.emit_inline_page_record(&addr, width);
        }
        if self.mode.traces_values() {
            let value = self.next_var();
            self.write_line(&format!("{cast_ty} {value} = ({cast_ty})({val});"));
            self.write_line(&format!("{trace_func}(state, {addr}, {value});"));
            self.write_line(&format!("{write_func}(memory, {addr}, {value});"));
        } else {
            self.write_line(&format!(
                "{write_func}(memory, {addr}, ({cast_ty})({val}));"
            ));
        }
    }

    fn emit_inline_page_record(&mut self, addr: &str, width: u8) {
        if width == 1 {
            self.write_line(&format!("trace_memory_access_leaf(&trace_memory, {addr});"));
        } else {
            self.write_line(&format!(
                "trace_memory_access_span(&trace_memory, {addr}, {width}u);"
            ));
        }
    }

    pub fn flush_page_locals(&mut self) {
        if self.mode.traces_memory_pages() {
            self.write_line("trace_memory_flush(&state->mode_state, &trace_memory);");
        }
    }

    pub fn reload_page_locals(&mut self) {
        if self.mode.traces_memory_pages() {
            self.write_line("trace_memory_reload(&state->mode_state, &trace_memory);");
        }
    }

    /// Emit a PC read when value tracing is enabled.
    pub fn trace_pc(&mut self, pc: u64) {
        if self.mode.traces_values() {
            self.write_line(&format!("trace_pc(state, 0x{pc:08x}ull);"));
        }
    }

    pub fn emit_call(&mut self, name: &str, args: &[&str]) {
        self.flush_page_locals();
        let args_str = args.join(", ");
        self.write_line(&format!("{name}({args_str});"));
        self.reload_page_locals();
    }

    pub fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        let args_str = args.join(", ");
        self.write_line(&format!("{name}({args_str});"));
    }

    pub fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
        self.flush_page_locals();
        let tmp = self.next_var();
        let args_str = args.join(", ");
        self.write_line(&format!("{ret_ty} {tmp} = {name}({args_str});"));
        self.reload_page_locals();
        tmp
    }

    pub fn emit_call_with_trace_result(
        &mut self,
        ret_ty: &str,
        name: &str,
        args: &[&str],
    ) -> Option<String> {
        if matches!(self.mode, EmitMode::ValueTrace | EmitMode::Direct) {
            self.emit_call(name, args);
            None
        } else {
            Some(self.emit_call_expr(ret_ty, name, args))
        }
    }

    pub fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
        if chip_idx == u32::MAX {
            return;
        }
        if !matches!(self.mode, EmitMode::ValueTrace | EmitMode::Direct) {
            let num_airs = self
                .num_airs
                .expect("metered code generation requires the AIR count");
            if let Err(error) = validate_chip_index(chip_idx, num_airs) {
                self.invalid_chip_index.get_or_insert(error);
                return;
            }
        }
        match self.mode {
            EmitMode::ValueTrace | EmitMode::Direct => {}
            EmitMode::Metered { .. } => {
                self.write_line(&format!("(*trace_heights)[{chip_idx}] += {count_expr};"));
            }
            EmitMode::MeteredCost => {
                let width = self
                    .chip_widths
                    .unwrap()
                    .get(chip_idx as usize)
                    .copied()
                    .expect("extension chip index exceeds chip-width table");
                if width != 0 {
                    self.write_line(&format!(
                        "state->mode_state.cost += {width}ull * (uint64_t)({count_expr});"
                    ));
                }
            }
        }
    }

    pub(crate) fn invalid_chip_index(&self) -> Option<InvalidChipIndex> {
        self.invalid_chip_index
    }

    pub fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str) {
        if chip_idx == u32::MAX || matches!(self.mode, EmitMode::ValueTrace | EmitMode::Direct) {
            return;
        }
        self.write_line(&format!("if (({count_expr}) != 0u) {{"));
        self.trace_chip(chip_idx, count_expr);
        self.write_line("}");
    }

    pub fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace) {
        assert!(
            !self.mode.traces_values(),
            "page-only access is invalid when values are being traced"
        );
        if !matches!(self.mode, EmitMode::Metered { .. }) {
            return;
        }
        let touches_memory = addr_space.is_main_memory();
        if touches_memory {
            self.flush_page_locals();
        }
        let size = width.bytes();
        self.write_line(&format!(
            "trace_page_access(state, {addr}, {size}u, {}u);",
            addr_space.id()
        ));
        if touches_memory {
            self.reload_page_locals();
        }
    }

    pub fn trace_page_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    ) {
        assert!(
            !self.mode.traces_values(),
            "page-only access is invalid when values are being traced"
        );
        if !matches!(self.mode, EmitMode::Metered { .. }) {
            return;
        }
        let touches_memory = addr_space.is_main_memory();
        if touches_memory {
            self.flush_page_locals();
        }
        self.write_line(&format!(
            "trace_page_access_u64_range(state, {base_addr}, {num_dwords}, {}u);",
            addr_space.id()
        ));
        if touches_memory {
            self.reload_page_locals();
        }
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
        match self.block_abi {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => args.push_str(", instret_remaining"),
            BlockAbi::Metered => args.push_str(", check_counter, trace_heights"),
        }
        args
    }

    pub fn sync_regs_to_state(&mut self) {
        self.flush_page_locals();
        let mut args = "state".to_string();
        for &idx in &self.sorted_hot_regs() {
            let name = Self::abi_name(idx);
            write!(args, ", {name}").unwrap();
        }
        self.write_line(&format!("rv_save_hot_regs({args});"));
        match self.block_abi {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => {
                self.write_line(
                    "state->mode_state.retired = state->mode_state.target - instret_remaining;",
                );
            }
            BlockAbi::Metered => {
                self.write_line("state->mode_state.check_counter = check_counter;");
            }
        }
    }

    pub fn sync_regs_from_state(&mut self) {
        for &idx in &self.sorted_hot_regs() {
            let name = Self::abi_name(idx);
            self.write_line(&format!("{name} = state->regs[{idx}];"));
        }
    }
}

impl rvr_openvm_ir::ExtEmitCtx for EmitContext<'_> {
    fn read_var(&mut self, var: Variable) -> String {
        EmitContext::read_reg(self, rv64_reg_index(var))
    }

    fn peek_var(&mut self, var: Variable) -> String {
        EmitContext::peek_reg(self, rv64_reg_index(var))
    }

    fn write_var(&mut self, var: Variable, val: &str) {
        EmitContext::write_reg(self, rv64_reg_index(var), val)
    }

    fn write_line(&mut self, s: &str) {
        EmitContext::write_line(self, s)
    }

    fn emit_trap(&mut self) {
        EmitContext::emit_trap(self)
    }

    fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        EmitContext::read_mem(self, base, offset, width, signed)
    }

    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        EmitContext::write_mem(self, base, offset, val, width);
    }

    fn emit_call(&mut self, name: &str, args: &[&str]) {
        EmitContext::emit_call(self, name, args);
    }

    fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        EmitContext::emit_call_without_page_flush(self, name, args);
    }

    fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
        EmitContext::emit_call_expr(self, ret_ty, name, args)
    }

    fn emit_call_with_trace_result(
        &mut self,
        ret_ty: &str,
        name: &str,
        args: &[&str],
    ) -> Option<String> {
        EmitContext::emit_call_with_trace_result(self, ret_ty, name, args)
    }

    fn trace_chip(&mut self, chip_idx: u32, count_expr: &str) {
        EmitContext::trace_chip(self, chip_idx, count_expr);
    }

    fn trace_chip_if_nonzero(&mut self, chip_idx: u32, count_expr: &str) {
        EmitContext::trace_chip_if_nonzero(self, chip_idx, count_expr);
    }

    fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace) {
        EmitContext::trace_page_access(self, addr, width, addr_space);
    }

    fn trace_page_access_u64_range(
        &mut self,
        base_addr: &str,
        num_dwords: &str,
        addr_space: PageAddressSpace,
    ) {
        EmitContext::trace_page_access_u64_range(self, base_addr, num_dwords, addr_space);
    }
}

fn rv64_reg_index(var: Variable) -> u8 {
    let index = u8::try_from(var.index()).expect("RV64 variable index must fit in u8");
    assert!(index < 32, "RV64 variable index must name x0..x31");
    index
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{BlockAbi, EmitContext, EmitMode};

    fn metered_memory_ctx() -> EmitContext<'static> {
        EmitContext::new(
            HashSet::new(),
            EmitMode::Metered {
                trace_memory_pages: true,
            },
            BlockAbi::Metered,
            None,
            Some(1),
        )
    }

    #[test]
    fn metered_memory_access_records_full_span() {
        let mut ctx = metered_memory_ctx();
        ctx.read_mem("addr", 0, 8, false);

        assert!(ctx
            .buf()
            .contains("trace_memory_access_span(&trace_memory, addr, 8u);"));
    }
}
