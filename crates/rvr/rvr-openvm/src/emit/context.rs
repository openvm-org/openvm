use std::{collections::HashSet, fmt::Write};

use openvm_instructions::MEMORY_BLOCK_BYTES;
use rvr_openvm_ir::{MemWidth, PageAddressSpace, Variable};
use rvr_state::NUM_REGS;

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
    /// Emit only block checkpoints and the replay values that execution
    /// cannot derive while replaying a block.
    Preflight,
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
    fn preserves_logical_schedule(self) -> bool {
        matches!(self, Self::Preflight)
    }

    fn uses_preflight_local(self) -> bool {
        matches!(self, Self::Preflight)
    }

    fn is_metered_without_memory_pages(self) -> bool {
        matches!(
            self,
            Self::Metered {
                trace_memory_pages: false
            }
        )
    }

    fn meters_replay_values(self) -> bool {
        matches!(self, Self::Metered { .. })
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

#[derive(Debug, Default, Eq, PartialEq)]
enum DynamicTimestampSlots {
    #[default]
    None,
    /// Capacity checked; the runtime expression has not been applied yet.
    Reserved(String),
    /// Runtime schedule applied; fixed slot accounting is suspended.
    Active,
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
    fixed_timestamp_slots: u32,
    fixed_replay_values: u32,
    dynamic_timestamp_slots: DynamicTimestampSlots,
    replay_values_reserved: bool,
    profile_execution: bool,
    current_pc: Option<u64>,
}

impl<'a> EmitContext<'a> {
    pub(crate) fn new(
        hot_regs: HashSet<u8>,
        mode: EmitMode,
        block_abi: BlockAbi,
        chip_widths: Option<&'a [u64]>,
        num_airs: Option<u32>,
        profile_execution: bool,
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
            fixed_timestamp_slots: 0,
            fixed_replay_values: 0,
            dynamic_timestamp_slots: DynamicTimestampSlots::None,
            replay_values_reserved: false,
            profile_execution,
            current_pc: None,
        }
    }

    pub(crate) fn preflight_block_budget(&self) -> (u32, u32) {
        assert!(
            self.dynamic_timestamp_slots == DynamicTimestampSlots::None
                && !self.replay_values_reserved,
            "unfinished preflight dynamic reservation"
        );
        (self.fixed_timestamp_slots, self.fixed_replay_values)
    }

    pub(crate) fn metered_block_replay_values(&self) -> u32 {
        if self.mode.meters_replay_values() {
            assert!(
                !self.replay_values_reserved,
                "unfinished metered replay-value reservation"
            );
            self.fixed_replay_values
        } else {
            0
        }
    }

    fn count_fixed_timestamp_slots(&mut self, slots: u32) {
        if !self.mode.uses_preflight_local()
            || self.dynamic_timestamp_slots == DynamicTimestampSlots::Active
        {
            return;
        }
        self.fixed_timestamp_slots = self
            .fixed_timestamp_slots
            .checked_add(slots)
            .expect("preflight block timestamp-slot count overflow");
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

    pub(crate) fn preserves_logical_schedule(&self) -> bool {
        self.mode.preserves_logical_schedule()
    }

    /// Account for logical memory slots not emitted through the standard
    /// memory-access helpers.
    pub(crate) fn advance_timestamp(&mut self, slots: u32) {
        if slots == 0 {
            return;
        }
        self.count_fixed_timestamp_slots(slots);
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
        self.flush_preflight_local();
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
            if matches!(kind, RegisterReadKind::MemoryAccess) {
                self.count_fixed_timestamp_slots(1);
            }
            return "0ull".to_string();
        }

        let value = if self.hot_regs.contains(&idx) {
            let name = Self::abi_name(idx);
            name.to_string()
        } else {
            let var = self.next_var();
            self.write_line(&format!(
                "[[maybe_unused]] uint64_t {var} = reg_read(state, {idx});"
            ));
            var
        };

        if matches!(kind, RegisterReadKind::MemoryAccess) {
            self.count_fixed_timestamp_slots(1);
        }

        value
    }

    /// Read a register as a VM memory access, emitting `trace_reg_read` in
    /// value-tracing mode.
    pub fn read_reg(&mut self, idx: u8) -> String {
        self.read_reg_impl(idx, RegisterReadKind::MemoryAccess)
    }

    /// Read a register at the current logical memory timestamp, emitting
    /// `trace_reg_peek` in value-tracing mode.
    pub fn peek_reg(&mut self, idx: u8) -> String {
        self.read_reg_impl(idx, RegisterReadKind::Peek)
    }

    /// Write a register as a VM memory access, emitting `trace_reg_write` in
    /// value-tracing mode.
    pub fn write_reg(&mut self, idx: u8, val: &str) {
        if idx == 0 {
            self.count_fixed_timestamp_slots(1);
            return;
        }
        self.count_fixed_timestamp_slots(1);
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

    fn read_mem_helper(width: u8, signed: bool) -> (&'static str, &'static str) {
        match (width, signed) {
            (1, false) => ("read_mem_u8", "uint32_t"),
            (1, true) => ("read_mem_i8", "int32_t"),
            (2, false) => ("read_mem_u16", "uint32_t"),
            (2, true) => ("read_mem_i16", "int32_t"),
            (4, false) => ("read_mem_u32", "uint32_t"),
            (4, true) => ("read_mem_i32", "int32_t"),
            (8, _) => ("read_mem_u64", "uint64_t"),
            _ => unreachable!("invalid memory width {width}"),
        }
    }

    fn write_mem_helper(width: u8) -> (&'static str, &'static str) {
        match width {
            1 => ("write_mem_u8", "uint8_t"),
            2 => ("write_mem_u16", "uint16_t"),
            4 => ("write_mem_u32", "uint32_t"),
            8 => ("write_mem_u64", "uint64_t"),
            _ => unreachable!("invalid memory width {width}"),
        }
    }

    /// Trap from the generated block before a protected scalar memory helper
    /// can reach its process-aborting backstop. This is the same predicate as
    /// the inlined helper check, so Clang folds the latter away; unprotected
    /// artifacts emit no check.
    #[cfg(not(feature = "unprotected"))]
    fn emit_memory_bounds_trap(&mut self, addr: &str, width: u8) {
        self.write_line(&format!(
            "if (unlikely((uint64_t)({addr}) > OPENVM_MEM_SIZE - {width}u)) {{"
        ));
        self.emit_trap();
        self.write_line("}");
    }

    #[cfg(feature = "unprotected")]
    fn emit_memory_bounds_trap(&mut self, _addr: &str, _width: u8) {}

    /// Read guest memory. Metered hot blocks record the memory page separately.
    fn read_mem_impl(
        &mut self,
        base: &str,
        offset: i16,
        width: u8,
        signed: bool,
        sp_relative: bool,
    ) -> String {
        assert!(
            !self.mode.is_metered_without_memory_pages(),
            "metered memory read emitted without page tracking"
        );
        let addr = Self::addr_expr(base, offset);
        let var = self.next_var();
        let (read_func, var_ty) = Self::read_mem_helper(width, signed);
        self.uses_raw_memory = true;

        self.emit_memory_bounds_trap(&addr, width);
        self.write_line(&format!("{var_ty} {var} = {read_func}(memory, {addr});"));
        self.count_fixed_timestamp_slots(if width == 1 { 1 } else { 2 });
        if self.mode.traces_memory_pages() {
            self.emit_inline_page_record(&addr, width, sp_relative);
        }
        var
    }

    pub fn read_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        self.read_mem_impl(base, offset, width, signed, false)
    }

    pub fn read_sp_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        self.read_mem_impl(base, offset, width, signed, true)
    }

    /// Emit a guest memory write. Metered hot blocks record the memory page
    /// through the block-local `TraceMemory` context, then use the raw memory
    /// helper so the common path avoids tracing calls.
    fn write_mem_impl(&mut self, base: &str, offset: i16, val: &str, width: u8, sp_relative: bool) {
        assert!(
            !self.mode.is_metered_without_memory_pages(),
            "metered memory write emitted without page tracking"
        );
        let addr = Self::addr_expr(base, offset);
        let (write_func, cast_ty) = Self::write_mem_helper(width);
        self.uses_raw_memory = true;

        self.emit_memory_bounds_trap(&addr, width);
        if self.mode.traces_memory_pages() {
            self.emit_inline_page_record(&addr, width, sp_relative);
        }
        self.count_fixed_timestamp_slots(if width == 1 { 1 } else { 2 });
        self.write_line(&format!(
            "{write_func}(memory, {addr}, ({cast_ty})({val}));"
        ));
        if self.mode.uses_preflight_local() {
            self.write_line(&format!(
                "preflight_local_mark_memory_write(state, &preflight, {addr}, {width}u);"
            ));
        }
    }

    pub fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        self.write_mem_impl(base, offset, val, width, false)
    }

    pub fn write_sp_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        self.write_mem_impl(base, offset, val, width, true)
    }

    /// Emit one naturally aligned main-memory block write.
    pub fn write_aligned_mem_block(&mut self, addr: &str, val: &str) {
        assert!(
            !self.mode.is_metered_without_memory_pages(),
            "metered memory block write emitted without page tracking"
        );
        self.uses_raw_memory = true;

        self.emit_memory_bounds_trap(addr, MEMORY_BLOCK_BYTES as u8);
        if self.mode.traces_memory_pages() {
            self.emit_inline_page_record(addr, MEMORY_BLOCK_BYTES as u8, false);
        }
        self.count_fixed_timestamp_slots(1);
        self.write_line(&format!(
            "write_mem_u64(memory, {addr}, (uint64_t)({val}));"
        ));
        if self.mode.uses_preflight_local() {
            self.write_line(&format!(
                "preflight_local_mark_memory_write(state, &preflight, {addr}, {MEMORY_BLOCK_BYTES}u);"
            ));
        }
    }

    /// Emit a fail-before-mutation capacity and timestamp-headroom check.
    pub fn reserve_preflight_timestamp_slots(&mut self, slots: &str) {
        if self.mode.uses_preflight_local() {
            assert!(
                self.dynamic_timestamp_slots == DynamicTimestampSlots::None,
                "nested preflight dynamic timestamp schedule"
            );
            self.write_line(&format!(
                "if (unlikely(!preflight_local_reserve(&preflight, 0u, {slots}))) {{"
            ));
            self.emit_trap();
            self.write_line("}");
            self.dynamic_timestamp_slots = DynamicTimestampSlots::Reserved(slots.to_string());
        }
    }

    pub fn reserve_replay_values(&mut self, count: &str) {
        if self.mode.meters_replay_values() {
            assert!(
                !self.replay_values_reserved,
                "nested metered replay-value reservation"
            );
            self.replay_values_reserved = true;
            self.emit_metered_replay_value_add(count);
            return;
        }
        if !self.mode.uses_preflight_local() {
            return;
        }
        assert!(
            !self.replay_values_reserved,
            "nested preflight replay-value reservation"
        );
        self.replay_values_reserved = true;
        self.write_line(&format!(
            "if (unlikely(!preflight_local_reserve_replay_values(&preflight, {count}))) {{"
        ));
        self.emit_trap();
        self.write_line("}");
        match std::mem::take(&mut self.dynamic_timestamp_slots) {
            DynamicTimestampSlots::Reserved(slots) => {
                self.write_line(&format!(
                    "preflight_local_add_timestamp_unchecked(&preflight, {slots});"
                ));
                self.dynamic_timestamp_slots = DynamicTimestampSlots::Active;
            }
            DynamicTimestampSlots::None => {}
            DynamicTimestampSlots::Active => {
                unreachable!("nested preflight dynamic timestamp schedule")
            }
        }
    }

    pub fn count_fixed_replay_values(&mut self, count: u32) {
        if !self.mode.meters_replay_values() {
            return;
        }
        assert!(
            !self.replay_values_reserved,
            "fixed replay-value count during a dynamic reservation"
        );
        self.fixed_replay_values = self
            .fixed_replay_values
            .checked_add(count)
            .expect("metered block replay-value count overflow");
    }

    pub fn count_replay_values(&mut self, count: &str) {
        if !self.mode.meters_replay_values() {
            return;
        }
        assert!(
            !self.replay_values_reserved,
            "dynamic replay-value count during a materialization reservation"
        );
        self.emit_metered_replay_value_add(count);
    }

    fn emit_metered_replay_value_add(&mut self, count: &str) {
        let count_var = self.next_var();
        self.write_line(&format!("uint64_t {count_var} = (uint64_t)({count});"));
        self.write_line(&format!(
            "if (unlikely({count_var} > (uint64_t)UINT32_MAX - state->mode_state.num_preflight_replay_values)) {{"
        ));
        self.emit_trap();
        self.write_line("}");
        self.write_line(&format!(
            "state->mode_state.num_preflight_replay_values += (uint32_t){count_var};"
        ));
    }

    pub fn append_replay_value(&mut self, value: &str) {
        if self.mode.meters_replay_values() {
            if self.replay_values_reserved {
                self.replay_values_reserved = false;
            } else {
                self.count_fixed_replay_values(1);
            }
            return;
        }
        if !self.mode.uses_preflight_local() {
            return;
        }
        if !self.replay_values_reserved {
            self.fixed_replay_values = self
                .fixed_replay_values
                .checked_add(1)
                .expect("preflight block replay-value count overflow");
        }
        self.write_line(&format!(
            "preflight_local_append_replay_value_unchecked(&preflight, (uint64_t)({value}));"
        ));
        self.replay_values_reserved = false;
        if self.dynamic_timestamp_slots == DynamicTimestampSlots::Active {
            self.dynamic_timestamp_slots = DynamicTimestampSlots::None;
        }
    }

    pub fn append_replay_memory_u64_range(&mut self, base: &str, count: &str) {
        if self.mode.meters_replay_values() {
            assert!(
                self.replay_values_reserved,
                "metered replay range requires a replay-value reservation"
            );
            self.replay_values_reserved = false;
            return;
        }
        if !self.mode.uses_preflight_local() {
            return;
        }
        assert!(
            self.replay_values_reserved,
            "preflight replay memory range requires a replay-value reservation"
        );
        self.write_line(&format!(
            "for (uint32_t replay_word = 0u; replay_word < {count}; ++replay_word) {{"
        ));
        self.append_replay_value(&format!(
            "peek_mem_u64(state, {base} + (uint64_t)replay_word * 8ull)"
        ));
        self.write_line("}");
    }

    fn emit_inline_page_record(&mut self, addr: &str, width: u8, sp_relative: bool) {
        let prefix = if sp_relative {
            "trace_sp_memory"
        } else {
            "trace_memory"
        };
        if width == 1 {
            self.write_line(&format!("{prefix}_access_leaf(&trace_memory, {addr});"));
        } else {
            self.write_line(&format!(
                "{prefix}_access_span(&trace_memory, {addr}, {width}u);"
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

    pub(crate) fn flush_preflight_local(&mut self) {
        if self.mode == EmitMode::Preflight {
            self.write_line("/* PREFLIGHT_FINISH_BLOCK */");
            self.write_line("preflight_local_flush(state, &preflight);");
        }
    }

    fn apply_reserved_timestamp_slots(&mut self) {
        let state = std::mem::take(&mut self.dynamic_timestamp_slots);
        match state {
            DynamicTimestampSlots::Reserved(slots) => {
                debug_assert!(self.mode.uses_preflight_local());
                self.write_line(&format!(
                    "preflight_local_add_timestamp_unchecked(&preflight, {slots});"
                ));
            }
            state => self.dynamic_timestamp_slots = state,
        }
    }

    /// Emit a PC read when value tracing is enabled.
    pub fn trace_pc(&mut self, pc: u64) {
        self.current_pc = Some(pc);
        if self.mode.traces_values() {
            self.write_line(&format!("trace_pc(state, 0x{pc:08x}ull);"));
        }
    }

    fn emit_profile_callsite(&mut self) {
        if self.profile_execution {
            let pc = self
                .current_pc
                .expect("external call emitted before instruction PC was recorded");
            self.write_line(&format!("rv_profile_guest_pc(state, 0x{pc:08x}ull);"));
        }
    }

    pub fn emit_call(&mut self, name: &str, args: &[&str]) {
        self.flush_page_locals();
        self.apply_reserved_timestamp_slots();
        self.emit_profile_callsite();
        let args_str = args.join(", ");
        self.write_line(&format!("{name}({args_str});"));
        self.reload_page_locals();
    }

    pub fn emit_call_without_page_flush(&mut self, name: &str, args: &[&str]) {
        self.emit_profile_callsite();
        let args_str = args.join(", ");
        self.write_line(&format!("{name}({args_str});"));
    }

    pub fn emit_call_expr(&mut self, ret_ty: &str, name: &str, args: &[&str]) -> String {
        self.flush_page_locals();
        self.apply_reserved_timestamp_slots();
        self.emit_profile_callsite();
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
        if matches!(self.mode, EmitMode::Preflight | EmitMode::Direct) {
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
        if !matches!(self.mode, EmitMode::Preflight | EmitMode::Direct) {
            let num_airs = self
                .num_airs
                .expect("metered code generation requires the AIR count");
            if let Err(error) = validate_chip_index(chip_idx, num_airs) {
                self.invalid_chip_index.get_or_insert(error);
                return;
            }
        }
        match self.mode {
            EmitMode::Preflight | EmitMode::Direct => {}
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
        if chip_idx == u32::MAX || matches!(self.mode, EmitMode::Preflight | EmitMode::Direct) {
            return;
        }
        self.write_line(&format!("if (({count_expr}) != 0u) {{"));
        self.trace_chip(chip_idx, count_expr);
        self.write_line("}");
    }

    pub fn trace_page_access(&mut self, addr: &str, width: MemWidth, addr_space: PageAddressSpace) {
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
        self.flush_preflight_local();
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
    fn is_preflight(&self) -> bool {
        self.mode.uses_preflight_local()
    }

    fn read_var(&mut self, var: Variable) -> String {
        EmitContext::read_reg(self, reg_index(var))
    }

    fn peek_var(&mut self, var: Variable) -> String {
        EmitContext::peek_reg(self, reg_index(var))
    }

    fn advance_timestamp(&mut self, slots: u32) {
        EmitContext::advance_timestamp(self, slots)
    }

    fn write_var(&mut self, var: Variable, val: &str) {
        EmitContext::write_reg(self, reg_index(var), val)
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

    fn read_sp_mem(&mut self, base: &str, offset: i16, width: u8, signed: bool) -> String {
        EmitContext::read_sp_mem(self, base, offset, width, signed)
    }

    fn write_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        EmitContext::write_mem(self, base, offset, val, width);
    }

    fn write_sp_mem(&mut self, base: &str, offset: i16, val: &str, width: u8) {
        EmitContext::write_sp_mem(self, base, offset, val, width);
    }

    fn write_aligned_mem_block(&mut self, addr: &str, val: &str) {
        EmitContext::write_aligned_mem_block(self, addr, val);
    }

    fn reserve_preflight_timestamp_slots(&mut self, slots: &str) {
        EmitContext::reserve_preflight_timestamp_slots(self, slots);
    }

    fn reserve_replay_values(&mut self, count: &str) {
        EmitContext::reserve_replay_values(self, count);
    }

    fn append_replay_value(&mut self, value: &str) {
        EmitContext::append_replay_value(self, value);
    }

    fn count_fixed_replay_values(&mut self, count: u32) {
        EmitContext::count_fixed_replay_values(self, count);
    }

    fn count_replay_values(&mut self, count: &str) {
        EmitContext::count_replay_values(self, count);
    }

    fn append_replay_memory_u64_range(&mut self, base: &str, count: &str) {
        EmitContext::append_replay_memory_u64_range(self, base, count);
    }

    fn flush_before_control_transfer(&mut self) {
        if self.mode == EmitMode::Preflight {
            self.write_line("preflight_local_flush(state, &preflight);");
        }
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

fn reg_index(var: Variable) -> u8 {
    let index = u8::try_from(var.index()).expect("variable index must fit in u8");
    assert!(
        usize::from(index) < NUM_REGS,
        "variable index must name a state register"
    );
    index
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rvr_openvm_ir::ExtEmitCtx;

    use super::{BlockAbi, EmitContext, EmitMode};

    fn preflight_ctx() -> EmitContext<'static> {
        EmitContext::new(
            HashSet::new(),
            EmitMode::Preflight,
            BlockAbi::Plain,
            None,
            None,
            false,
        )
    }

    #[test]
    fn advance_timestamp_counts_fixed_preflight_slots() {
        let mut preflight = preflight_ctx();
        preflight.advance_timestamp(3);
        assert!(preflight.buf().is_empty());
        assert_eq!(preflight.preflight_block_budget(), (3, 0));

        for mode in [
            EmitMode::Direct,
            EmitMode::Metered {
                trace_memory_pages: false,
            },
            EmitMode::MeteredCost,
        ] {
            let chip_widths = matches!(mode, EmitMode::MeteredCost).then_some(&[][..]);
            let block_abi = if matches!(mode, EmitMode::Metered { .. }) {
                BlockAbi::Metered
            } else {
                BlockAbi::Plain
            };
            let mut ctx =
                EmitContext::new(HashSet::new(), mode, block_abi, chip_widths, Some(0), false);
            ctx.advance_timestamp(3);
            assert!(ctx.buf().is_empty());
        }
    }

    #[test]
    fn control_transfer_flush_preserves_pure_and_metered_codegen() {
        let mut preflight = preflight_ctx();
        preflight.flush_before_control_transfer();
        assert_eq!(
            preflight.buf(),
            "        preflight_local_flush(state, &preflight);\n"
        );
        assert!(!preflight.buf().contains("PREFLIGHT_FINISH_BLOCK"));

        for mode in [
            EmitMode::Direct,
            EmitMode::Metered {
                trace_memory_pages: false,
            },
            EmitMode::Metered {
                trace_memory_pages: true,
            },
            EmitMode::MeteredCost,
        ] {
            let chip_widths = matches!(mode, EmitMode::MeteredCost).then_some(&[][..]);
            let block_abi = if matches!(mode, EmitMode::Metered { .. }) {
                BlockAbi::Metered
            } else {
                BlockAbi::Plain
            };
            let mut ctx =
                EmitContext::new(HashSet::new(), mode, block_abi, chip_widths, Some(0), false);
            ctx.flush_before_control_transfer();
            assert!(ctx.buf().is_empty(), "mode {mode:?} changed codegen");
        }
    }

    #[test]
    fn preflight_counts_schedule_without_access_events() {
        let mut ctx = preflight_ctx();
        assert_eq!(ctx.read_reg(0), "0ull");
        assert_eq!(ctx.peek_reg(0), "0ull");
        assert_eq!(ctx.read_reg(3), "_v0");
        ctx.write_reg(0, "1ull");
        ctx.write_reg(4, "2ull");
        ctx.read_mem("addr", 0, 8, false);
        ctx.write_mem("addr", 0, "3ull", 1);
        ctx.write_aligned_mem_block("addr", "4ull");
        ctx.advance_timestamp(3);

        assert_eq!(ctx.preflight_block_budget(), (11, 0));
        assert!(!ctx.buf().contains("preflight_local_reg"));
        assert!(!ctx.buf().contains("trace_read"));
        assert!(!ctx.buf().contains("trace_write"));
        assert!(!ctx.buf().contains("trace_reg"));
        assert_eq!(
            ctx.buf()
                .matches("preflight_local_mark_memory_write")
                .count(),
            2
        );
    }

    #[test]
    fn preflight_replay_values_are_untagged_and_fixed_by_default() {
        let mut ctx = preflight_ctx();
        ctx.append_replay_value("loaded");

        assert_eq!(ctx.preflight_block_budget(), (0, 1));
        assert_eq!(
            ctx.buf(),
            "        preflight_local_append_replay_value_unchecked(&preflight, (uint64_t)(loaded));\n"
        );
    }

    #[test]
    fn metered_replay_values_fold_fixed_counts_and_emit_dynamic_counts() {
        let mut ctx = metered_memory_ctx();
        ctx.count_fixed_replay_values(4);
        ctx.append_replay_value("fixed");
        ctx.reserve_replay_values("words");
        ctx.append_replay_memory_u64_range("buffer", "words");
        ctx.count_replay_values("late_words");

        assert_eq!(ctx.metered_block_replay_values(), 5);
        assert!(ctx.buf().contains("uint64_t _v0 = (uint64_t)(words);"));
        assert!(ctx.buf().contains(
            "_v0 > (uint64_t)UINT32_MAX - state->mode_state.num_preflight_replay_values"
        ));
        assert!(ctx.buf().contains("uint64_t _v1 = (uint64_t)(late_words);"));
        assert!(ctx.buf().contains(
            "_v1 > (uint64_t)UINT32_MAX - state->mode_state.num_preflight_replay_values"
        ));
        assert_eq!(ctx.buf().matches("return rv_trap(").count(), 2);
    }

    #[test]
    fn preflight_dynamic_hint_schedule_and_replay_values_are_counted_once() {
        let mut ctx = preflight_ctx();
        ctx.append_replay_value("before");
        ctx.reserve_preflight_timestamp_slots("slots");
        ctx.reserve_replay_values("count");
        ctx.advance_timestamp(2);
        ctx.write_aligned_mem_block("addr", "hint");
        ctx.append_replay_value("hint");
        ctx.append_replay_value("after");
        ctx.read_reg(3);

        assert_eq!(ctx.preflight_block_budget(), (1, 2));
        assert_eq!(
            ctx.buf()
                .matches("preflight_local_add_timestamp_unchecked")
                .count(),
            1
        );
        assert_eq!(
            ctx.buf()
                .matches("preflight_local_reserve_replay_values")
                .count(),
            1
        );
        assert_eq!(
            ctx.buf()
                .matches("preflight_local_append_replay_value_unchecked")
                .count(),
            3
        );
        assert_eq!(
            ctx.buf()
                .matches(
                    "preflight_local_append_replay_value_unchecked(&preflight, (uint64_t)(hint));"
                )
                .count(),
            1
        );
        assert!(ctx
            .buf()
            .contains("preflight_local_reserve(&preflight, 0u, slots)"));
    }

    #[test]
    fn replay_memory_range_materializes_only_in_preflight_and_counts_in_metered() {
        let mut preflight = preflight_ctx();
        preflight.reserve_replay_values("words");
        preflight.append_replay_memory_u64_range("buffer", "words");

        assert_eq!(preflight.preflight_block_budget(), (0, 0));
        assert!(preflight
            .buf()
            .contains("for (uint32_t replay_word = 0u; replay_word < words; ++replay_word) {"));
        assert!(preflight
            .buf()
            .contains("peek_mem_u64(state, buffer + (uint64_t)replay_word * 8ull)"));

        let mut metered = EmitContext::new(
            HashSet::new(),
            EmitMode::Metered {
                trace_memory_pages: false,
            },
            BlockAbi::Metered,
            None,
            Some(0),
            false,
        );
        metered.reserve_replay_values("words");
        metered.append_replay_memory_u64_range("buffer", "words");
        assert!(metered.buf().contains("uint64_t _v0 = (uint64_t)(words);"));
        assert!(metered
            .buf()
            .contains("state->mode_state.num_preflight_replay_values += (uint32_t)_v0;"));

        for mode in [EmitMode::Direct, EmitMode::MeteredCost] {
            let chip_widths = matches!(mode, EmitMode::MeteredCost).then_some(&[][..]);
            let block_abi = if matches!(mode, EmitMode::Metered { .. }) {
                BlockAbi::Metered
            } else {
                BlockAbi::Plain
            };
            let mut ctx =
                EmitContext::new(HashSet::new(), mode, block_abi, chip_widths, Some(0), false);
            ctx.reserve_replay_values("words");
            ctx.append_replay_memory_u64_range("buffer", "words");
            assert!(ctx.buf().is_empty());
        }
    }

    #[test]
    fn preflight_host_call_does_not_finish_the_block() {
        let mut ctx = preflight_ctx();
        ctx.emit_call("host_call", &["state"]);
        assert_eq!(ctx.buf(), "        host_call(state);\n");

        ctx.flush_preflight_local();
        assert!(ctx.buf().contains("/* PREFLIGHT_FINISH_BLOCK */"));
        assert!(ctx
            .buf()
            .contains("preflight_local_flush(state, &preflight);"));
    }

    #[test]
    fn preflight_host_call_consumes_reserved_slots_without_fixed_budget() {
        let mut ctx = preflight_ctx();
        ctx.reserve_preflight_timestamp_slots("2u");
        let result = ctx.emit_call_expr("bool", "host_call", &["state"]);

        assert_eq!(result, "_v0");
        assert_eq!(ctx.preflight_block_budget(), (0, 0));
        let reserve = ctx
            .buf()
            .find("preflight_local_reserve(&preflight, 0u, 2u)")
            .unwrap();
        let advance = ctx
            .buf()
            .find("preflight_local_add_timestamp_unchecked(&preflight, 2u)")
            .unwrap();
        let call = ctx.buf().find("bool _v0 = host_call(state);").unwrap();
        assert!(reserve < advance && advance < call);
    }

    #[test]
    fn aligned_block_write_is_an_ordinary_raw_write_outside_preflight() {
        let mut ctx = EmitContext::new(
            HashSet::new(),
            EmitMode::Direct,
            BlockAbi::Plain,
            None,
            Some(0),
            false,
        );
        ctx.write_aligned_mem_block("addr", "value");
        ctx.reserve_preflight_timestamp_slots("13u");

        #[cfg(not(feature = "unprotected"))]
        {
            assert!(ctx
                .buf()
                .contains("if (unlikely((uint64_t)(addr) > OPENVM_MEM_SIZE - 8u)) {"));
            assert!(ctx.buf().contains("return rv_trap(state);"));
        }
        assert!(ctx
            .buf()
            .contains("write_mem_u64(memory, addr, (uint64_t)(value));"));
    }

    #[test]
    #[cfg(not(feature = "unprotected"))]
    fn protected_scalar_memory_traps_before_the_raw_access() {
        let mut ctx = EmitContext::new(
            HashSet::new(),
            EmitMode::Direct,
            BlockAbi::Plain,
            None,
            Some(0),
            false,
        );
        ctx.read_mem("addr", 3, 4, false);

        let bounds = ctx
            .buf()
            .find("if (unlikely((uint64_t)(addr + 0x00000003u) > OPENVM_MEM_SIZE - 4u)) {")
            .expect("protected bounds guard");
        let trap = ctx
            .buf()
            .find("return rv_trap(state);")
            .expect("typed trap");
        let read = ctx
            .buf()
            .find("read_mem_u32(memory, addr + 0x00000003u)")
            .expect("raw read");
        assert!(bounds < trap && trap < read);
    }

    #[test]
    fn metered_aligned_block_write_records_pages_once() {
        let mut ctx = metered_memory_ctx();
        ctx.write_aligned_mem_block("addr", "value");

        assert_eq!(
            ctx.buf()
                .matches("trace_memory_access_span(&trace_memory, addr, 8u);")
                .count(),
            1
        );
        assert_eq!(ctx.buf().matches("write_mem_u64(").count(), 1);
        assert!(!ctx.buf().contains("trace_page_access("));
    }

    fn metered_memory_ctx() -> EmitContext<'static> {
        EmitContext::new(
            HashSet::new(),
            EmitMode::Metered {
                trace_memory_pages: true,
            },
            BlockAbi::Metered,
            None,
            Some(1),
            false,
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

    #[test]
    fn metered_sp_access_uses_the_sp_relative_cache() {
        let mut ctx = metered_memory_ctx();
        ctx.read_sp_mem("sp", 4, 1, false);
        ctx.write_sp_mem("sp", -8, "value", 8);

        assert!(ctx
            .buf()
            .contains("trace_sp_memory_access_leaf(&trace_memory, sp + 0x00000004u);"));
        assert!(ctx
            .buf()
            .contains("trace_sp_memory_access_span(&trace_memory, sp - 0x00000008u, 8u);"));
        assert!(!ctx.buf().contains("trace_memory_access_leaf(&trace_memory"));
        assert!(!ctx.buf().contains("trace_memory_access_span(&trace_memory"));
    }

    fn external_call_output(profile_execution: bool) -> String {
        let mut ctx = EmitContext::new(
            HashSet::new(),
            EmitMode::Direct,
            BlockAbi::Plain,
            None,
            None,
            profile_execution,
        );
        ctx.trace_pc(0x20_1234);
        ctx.emit_call_without_page_flush("host_callback", &["value"]);
        ctx.take_buf()
    }

    #[test]
    fn profiling_changes_external_call_only_by_an_ordered_snapshot() {
        let unprofiled = external_call_output(false);
        let profiled = external_call_output(true);

        assert_eq!(unprofiled, "        host_callback(value);\n");
        assert_eq!(
            profiled,
            concat!(
                "        rv_profile_guest_pc(state, 0x00201234ull);\n",
                "        host_callback(value);\n",
            )
        );
    }
}
