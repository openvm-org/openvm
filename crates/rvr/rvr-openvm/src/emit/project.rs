//! C project generation: header, block files, dispatch, Makefile.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write,
    fs, io,
    path::{Path, PathBuf},
};

use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::*;
use rvr_openvm_lift::{ExtensionRegistry, TraceChipIndex};

use super::{
    codegen::{emit_terminator, InstrCodegen, TermCtx},
    context::EmitContext,
};

/// Compile-time tracer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracerMode {
    /// Pure execution with optional instruction-count suspension.
    Pure,
    /// Scalar metered cost accumulation (MeteredCostCtx).
    MeteredCost,
    /// Per-chip trace heights (MeteredCtx).
    Metered,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuspendPolicy {
    Disabled,
    InstretLimit,
    SegmentBoundary,
}

impl TracerMode {
    /// The tracer header filename (without directory).
    pub fn header_filename(self) -> &'static str {
        match self {
            TracerMode::Pure => "openvm_tracer_pure.h",
            TracerMode::MeteredCost => "openvm_tracer_metered_cost.h",
            TracerMode::Metered => "openvm_tracer_metered.h",
        }
    }

    fn header_content(self) -> &'static str {
        match self {
            TracerMode::Pure => include_str!("../../c/tracer/openvm_tracer_pure.h"),
            TracerMode::MeteredCost => {
                include_str!("../../c/tracer/openvm_tracer_metered_cost.h")
            }
            TracerMode::Metered => include_str!("../../c/tracer/openvm_tracer_metered.h"),
        }
    }

    pub fn default_suspend_policy(self) -> SuspendPolicy {
        match self {
            TracerMode::Pure | TracerMode::MeteredCost => SuspendPolicy::InstretLimit,
            TracerMode::Metered => SuspendPolicy::Disabled,
        }
    }

    fn block_header_content(self, suspend_policy: SuspendPolicy) -> &'static str {
        match (self, suspend_policy) {
            (TracerMode::Metered, SuspendPolicy::SegmentBoundary) => {
                include_str!("../../c/block/openvm_block_metered_segment.h")
            }
            (TracerMode::Metered, _) => include_str!("../../c/block/openvm_block_metered.h"),
            (TracerMode::Pure | TracerMode::MeteredCost, _) => {
                include_str!("../../c/block/openvm_block_instret.h")
            }
        }
    }
}

impl SuspendPolicy {
    fn header_content(self) -> &'static str {
        match self {
            SuspendPolicy::Disabled => include_str!("../../c/suspender/openvm_suspender_none.h"),
            SuspendPolicy::InstretLimit => {
                include_str!("../../c/suspender/openvm_suspender_instret_limit.h")
            }
            SuspendPolicy::SegmentBoundary => {
                include_str!("../../c/suspender/openvm_suspender_segment_boundary.h")
            }
        }
    }
}

/// C project generator.
pub struct CProject {
    output_dir: PathBuf,
    name: String,
    pub tracer_mode: TracerMode,
    pub suspend_policy: SuspendPolicy,
    pub hot_regs: HashSet<u8>,
    /// Maximum blocks per partition file.
    pub blocks_per_partition: usize,
    /// Enable thin LTO for the generated C code.
    pub enable_lto: bool,
    /// Per-PC chip index for hardcoded trace_chip calls.
    /// Index i = chip for PC = pc_base + i*4.
    /// `None` in pure mode (no chip metadata requested); must be set in metered modes.
    pub pc_to_chip: Option<Vec<TraceChipIndex>>,
    /// Program PC base (used to compute pc_to_chip index).
    pub pc_base: u32,
    /// Per-AIR widths for MeteredCost precomputation. Indexed by chip index.
    pub chip_widths: Option<Vec<u64>>,
    /// Compile with native debug info (`-g -fno-omit-frame-pointer`).
    pub native_debug_info: bool,
}

impl CProject {
    pub fn new(output_dir: &Path, name: &str, tracer_mode: TracerMode) -> Self {
        // Hot registers in priority order (matching original rvr's REG_PRIORITY).
        // Limited by platform's preserve_none register capacity minus 1 (state ptr).
        let hot_regs = Self::default_hot_regs();

        Self {
            output_dir: output_dir.to_path_buf(),
            name: name.to_string(),
            tracer_mode,
            suspend_policy: tracer_mode.default_suspend_policy(),
            hot_regs,
            blocks_per_partition: 512,
            enable_lto: true,
            pc_to_chip: None,
            pc_base: 0,
            chip_widths: None,
            native_debug_info: false,
        }
    }

    /// Register priority order (matching original rvr's REG_PRIORITY).
    /// x0 (zero) is excluded since it's always 0.
    const REG_PRIORITY: [u8; 31] = [
        1, 2, // ra, sp
        10, 11, 12, 13, 14, 15, 16, 17, // a0-a7
        5, 6, 7, // t0-t2
        28, 29, 30, 31, // t3-t6
        8, 9, // s0-s1
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, // s2-s11
        3, 4, // gp, tp
    ];

    /// Default hot registers, limited by platform's preserve_none register capacity.
    ///
    /// x86_64: 11 total preserve_none slots minus 1 (state) = 10 hot regs.
    /// aarch64: 24 total preserve_none slots minus 1 (state) = 23 hot regs.
    fn default_hot_regs() -> HashSet<u8> {
        #[cfg(target_arch = "aarch64")]
        const NUM_HOT_REGS: usize = 23;
        #[cfg(not(target_arch = "aarch64"))]
        const NUM_HOT_REGS: usize = 10;

        Self::REG_PRIORITY[..NUM_HOT_REGS].iter().copied().collect()
    }

    /// Sorted hot registers for deterministic signatures.
    fn sorted_hot_regs(&self) -> Vec<(u8, &'static str)> {
        let mut regs: Vec<(u8, &'static str)> = self
            .hot_regs
            .iter()
            .copied()
            .map(|r| (r, EmitContext::abi_name_static(r)))
            .collect();
        regs.sort_by_key(|&(idx, _)| idx);
        regs
    }

    /// Single-line C function parameter list.
    fn param_list(&self) -> String {
        let mut s = "RvState* restrict state".to_string();
        for &(_, name) in &self.sorted_hot_regs() {
            write!(s, ", uint32_t {name}").unwrap();
        }
        s
    }

    /// C argument list extracting hot regs from state:
    /// "state, state->regs[1], state->regs[2]".
    fn fn_args_from_state(&self) -> String {
        let mut s = "state".to_string();
        for &(idx, _) in &self.sorted_hot_regs() {
            write!(s, ", state->regs[{idx}]").unwrap();
        }
        s
    }

    /// C typedef parameter types: "RvState*, uint32_t, uint32_t".
    fn typedef_params(&self) -> String {
        let mut s = "RvState* restrict".to_string();
        for _ in &self.sorted_hot_regs() {
            s.push_str(", uint32_t");
        }
        s
    }

    /// Call to save hot registers back to state before returning.
    fn save_hot_regs_call(&self) -> String {
        let mut s = "rv_save_hot_regs(state".to_string();
        for &(_, name) in &self.sorted_hot_regs() {
            write!(s, ", {name}").unwrap();
        }
        s.push_str(");");
        s
    }

    fn escape_c_string_literal(value: &str) -> String {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    }

    fn block_end_pc(&self, block: &Block) -> u32 {
        block.terminator_pc.saturating_add(4)
    }
    /// Look up the chip index for a given PC. Must only be called in metered
    /// modes; panics if `pc_to_chip` is unset.
    fn chip_idx_for_pc(&self, pc: u32) -> TraceChipIndex {
        let mapping = self
            .pc_to_chip
            .as_ref()
            .expect("pc_to_chip must be set for metered rvr codegen");
        let Some(offset) = pc.checked_sub(self.pc_base) else {
            return TraceChipIndex::NoChip;
        };
        mapping
            .get((offset / 4) as usize)
            .copied()
            .unwrap_or(TraceChipIndex::NoChip)
    }

    /// Write all C project files.
    pub fn write_all<F: PrimeField32>(
        &self,
        blocks: &[Block],
        entry_point: u32,
        text_start: u32,
        extensions: &ExtensionRegistry<F>,
    ) -> io::Result<()> {
        self.write_constants()?;
        self.write_support_files()?;
        self.write_extension_files(extensions)?;
        let ext_headers = extensions.c_headers();
        self.write_header(blocks, text_start, &ext_headers)?;
        self.write_block_files(blocks)?;
        self.write_dispatch(blocks, entry_point, text_start)?;
        self.write_makefile()?;
        Ok(())
    }

    // ── Generated constants header ──────────────────────────────────────

    fn write_constants(&self) -> io::Result<()> {
        let h = crate::constants::constants_header();
        let path = self.output_dir.join("openvm_constants.h");
        fs::write(&path, h)
    }

    // ── Support files (tracer header, state header, IO) ─────────────────

    fn write_support_files(&self) -> io::Result<()> {
        // RvState struct definition (forward-declares Tracer).
        let state_path = self.output_dir.join("openvm_state.h");
        fs::write(&state_path, include_str!("../../c/openvm_state.h"))?;

        // Tracer header (includes openvm_state.h, defines Tracer + inline functions).
        let tracer_path = self.output_dir.join(self.tracer_mode.header_filename());
        fs::write(&tracer_path, self.tracer_mode.header_content())?;

        // Block and suspender headers are selected like tracer headers, then
        // copied under stable include names so generated block C does not
        // depend on mode or policy.
        let block_path = self.output_dir.join("openvm_block.h");
        fs::write(
            &block_path,
            self.tracer_mode.block_header_content(self.suspend_policy),
        )?;
        let suspender_path = self.output_dir.join("openvm_suspender.h");
        fs::write(&suspender_path, self.suspend_policy.header_content())?;

        // RISC-V M-extension helpers.
        let muldiv_path = self.output_dir.join("rv_muldiv.h");
        fs::write(&muldiv_path, include_str!("../../c/rv_muldiv.h"))?;

        // Memory-bounds check header declares per-width helpers;
        let bounds_h_path = self.output_dir.join("openvm_check_mem_bounds.h");
        fs::write(&bounds_h_path, include_str!("../../c/openvm_check_mem_bounds.h"))?;
        
        // Memory-bounds check implementation depending on protected/unprotected feature;
        let bounds_c_path = self.output_dir.join("openvm_check_mem_bounds.c");
        let content = include_str!("../../c/openvm_check_mem_bounds_protected.c");
        #[cfg(feature = "unprotected")]
        let content = include_str!("../../c/openvm_check_mem_bounds_unprotected.c");
        fs::write(&bounds_c_path, content)?;

        // IO implementation.
        fs::write(
            self.output_dir.join("openvm_io.c"),
            include_str!("../../c/openvm_io.c"),
        )?;
        fs::write(
            self.output_dir.join("openvm_io.h"),
            include_str!("../../c/openvm_io.h"),
        )?;

        let mut openvm_h = String::new();
        writeln!(
            openvm_h,
            "#include \"{}\"",
            self.tracer_mode.header_filename()
        )
        .unwrap();
        writeln!(openvm_h, "#include \"openvm_block.h\"").unwrap();
        writeln!(openvm_h, "#include \"openvm_suspender.h\"").unwrap();
        writeln!(openvm_h, "#include \"openvm_io.h\"").unwrap();
        fs::write(self.output_dir.join("openvm.h"), openvm_h)?;

        Ok(())
    }

    // ── Extension files ─────────────────────────────────────────────────

    fn write_extension_files<F: PrimeField32>(
        &self,
        extensions: &ExtensionRegistry<F>,
    ) -> io::Result<()> {
        let mut created_dirs = HashSet::new();
        self.write_embedded_files(extensions.c_headers(), &mut created_dirs)?;
        self.write_embedded_files(extensions.c_sources(), &mut created_dirs)?;
        self.write_embedded_files(extensions.extra_c_sources(), &mut created_dirs)?;
        self.write_embedded_files(extensions.extra_c_include_files(), &mut created_dirs)?;

        // Write wrapper functions if any extensions are registered
        if !extensions.is_empty() {
            self.write_ext_wrappers()?;
        }

        Ok(())
    }

    fn write_embedded_files(
        &self,
        files: impl IntoIterator<Item = (&'static str, &'static str)>,
        created_dirs: &mut HashSet<PathBuf>,
    ) -> io::Result<()> {
        for (filename, content) in files {
            let path = self.output_dir.join(filename);
            if let Some(parent) = path.parent() {
                let parent = parent.to_path_buf();
                if created_dirs.insert(parent.clone()) {
                    fs::create_dir_all(parent)?;
                }
            }
            fs::write(&path, content)?;
        }
        Ok(())
    }

    fn write_ext_wrappers(&self) -> io::Result<()> {
        fs::write(
            self.output_dir.join("rvr_ext_wrappers.c"),
            include_str!("../../c/rvr_ext_wrappers.c"),
        )?;
        Ok(())
    }

    // ── Main header ──────────────────────────────────────────────────────

    fn write_header(
        &self,
        blocks: &[Block],
        text_start: u32,
        ext_headers: &[(&str, &str)],
    ) -> io::Result<()> {
        let name = &self.name;
        let tracer_header = self.tracer_mode.header_filename();
        let mut h = String::with_capacity(4096);

        // Tracer header (includes openvm_state.h internally).
        writeln!(h, "#include \"{tracer_header}\"").unwrap();
        writeln!(h, "#include \"openvm_block.h\"").unwrap();
        writeln!(h, "#include \"openvm_suspender.h\"").unwrap();
        writeln!(h).unwrap();

        // Block function type and dispatch table (for JumpDyn tail calls).
        let typedef_params = self.typedef_params();
        writeln!(
            h,
            "typedef __attribute__((preserve_none)) void (*BlockFn)({typedef_params});"
        )
        .unwrap();
        writeln!(h, "extern BlockFn dispatch_table[];").unwrap();
        writeln!(h).unwrap();
        writeln!(
            h,
            "static __attribute__((always_inline)) inline uint32_t dispatch_index(uint32_t pc) {{"
        )
        .unwrap();
        writeln!(h, "    return (pc - 0x{text_start:08x}u) >> 2;").unwrap();
        writeln!(h, "}}").unwrap();
        writeln!(h).unwrap();

        // Block function declarations.
        let params = self.param_list();
        for block in blocks {
            writeln!(
                h,
                "__attribute__((preserve_none)) void block_0x{:08x}({params});",
                block.start_pc
            )
            .unwrap();
        }
        writeln!(h).unwrap();

        writeln!(
            h,
            "static __attribute__((always_inline)) inline void rv_save_hot_regs({params}) {{"
        )
        .unwrap();
        for &(idx, name) in &self.sorted_hot_regs() {
            writeln!(h, "    state->regs[{idx}] = {name};").unwrap();
        }
        writeln!(h, "}}").unwrap();
        writeln!(h).unwrap();

        writeln!(
            h,
            "/* rv_execute mutates state/tracer in place; Rust infers termination"
        )
        .unwrap();
        writeln!(
            h,
            " * or suspension from that state instead of a parallel status ABI. */"
        )
        .unwrap();
        writeln!(h).unwrap();

        // M-extension and IO headers.
        writeln!(h, "#include \"rv_muldiv.h\"").unwrap();
        writeln!(h, "#include \"openvm_io.h\"").unwrap();
        for &(filename, _) in ext_headers {
            writeln!(h, "#include \"{filename}\"").unwrap();
        }
        writeln!(h).unwrap();
        writeln!(
            h,
            "__attribute__((hot, nonnull)) void rv_execute(RvState* restrict state);"
        )
        .unwrap();

        let path = self.output_dir.join(format!("{name}.h"));
        fs::write(&path, h)
    }

    // ── Block files ──────────────────────────────────────────────────────

    fn write_block_files(&self, blocks: &[Block]) -> io::Result<()> {
        let name = &self.name;
        let num_partitions = blocks.len().div_ceil(self.blocks_per_partition);

        // Precompute valid block PCs for tail-call target validation.
        let valid_blocks: HashSet<u32> = blocks.iter().map(|b| b.start_pc).collect();

        for part_idx in 0..num_partitions {
            let start = part_idx * self.blocks_per_partition;
            let end = (start + self.blocks_per_partition).min(blocks.len());
            let partition = &blocks[start..end];

            let first_pc = partition[0].start_pc;
            let mut src = String::with_capacity(64 * 1024);
            writeln!(src, "#include \"{name}.h\"").unwrap();
            writeln!(src).unwrap();

            for block in partition {
                self.emit_block_function(&mut src, block, &valid_blocks);
            }

            let path = self.output_dir.join(format!("{name}_0x{first_pc:08x}.c"));
            fs::write(&path, src)?;
        }

        Ok(())
    }

    fn emit_block_function(&self, out: &mut String, block: &Block, valid_blocks: &HashSet<u32>) {
        let pc = block.start_pc;
        let end_pc = self.block_end_pc(block);
        let insn_count = block.insn_count();
        let params = self.param_list();
        let save = self.save_hot_regs_call();

        // Build tail-call argument list: "state, ra, sp".
        let mut args = "state".to_string();
        for &(_, name) in &self.sorted_hot_regs() {
            write!(args, ", {name}").unwrap();
        }

        writeln!(
            out,
            "// Block: 0x{pc:08x}-0x{:08x} ({insn_count} instrs)",
            end_pc.saturating_sub(1)
        )
        .unwrap();
        writeln!(
            out,
            "__attribute__((preserve_none)) void block_0x{pc:08x}({params}) {{"
        )
        .unwrap();

        // Body instructions (each in its own scope to avoid variable collisions).
        let mut ctx = EmitContext::new(self.hot_regs.clone());

        writeln!(
            out,
            "    uint8_t suspend_signal = begin_block(state, 0x{pc:08x}u, {insn_count}u);"
        )
        .unwrap();
        writeln!(
            out,
            "    if (unlikely(should_suspend(state, 0x{pc:08x}u, {insn_count}u, suspend_signal))) {{"
        )
        .unwrap();
        writeln!(out, "        state->has_exited = OPENVM_EXEC_SUSPENDED;").unwrap();
        writeln!(out, "        state->exit_code = 0;").unwrap();
        writeln!(out, "        {save}").unwrap();
        writeln!(out, "        state->pc = 0x{pc:08x}u;").unwrap();
        writeln!(out, "        return;").unwrap();
        writeln!(out, "    }}").unwrap();

        // Per-block chip accounting: batch all per-instruction chip
        // contributions into a single tracer update at block entry.
        self.emit_per_block_chip_updates(out, block);

        for instr_at in &block.instructions {
            self.emit_source_annotation(
                out,
                instr_at.pc,
                instr_at.instr.opname(),
                instr_at.source_loc.as_ref(),
            );
            ctx.trace_pc(instr_at.pc);
            instr_at.instr.emit_c(&mut ctx);
            out.push_str("    {\n");
            out.push_str(&ctx.take_buf());
            out.push_str("    }\n\n");
        }

        // Terminator (emits tail calls, no save-back needed for normal flow).
        let tc = TermCtx { valid_blocks };
        if !matches!(block.terminator, Terminator::FallThrough) {
            self.emit_source_annotation(
                out,
                block.terminator_pc,
                block.terminator.opname(),
                block.terminator_source_loc.as_ref(),
            );
            ctx.trace_pc(block.terminator_pc);
        }
        emit_terminator(&mut ctx, &block.terminator, block.terminator_pc, &tc);
        out.push_str("    {\n");
        out.push_str(&ctx.take_buf());
        out.push_str("    }\n");

        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    /// Sum the chip contributions for every instruction in `block` (body
    /// instructions plus the real terminator instruction, if any) and emit a
    /// batched tracer update instead of per-instruction `trace_chip` calls.
    ///
    /// Mode-dependent emission:
    ///   - Pure: nothing (chip tracking is a no-op).
    ///   - Metered: `_trace_heights[idx] += count;` per distinct chip, against a local cached from
    ///     `state->tracer->trace_heights` so each update is a single indexed store instead of a
    ///     chained load through `state->` and `state->tracer->`.
    ///   - MeteredCost: `state->tracer->cost += <constant>;` where the constant is `sum(width[chip]
    ///     * count)` precomputed at emit time from `self.chip_widths`.
    fn emit_per_block_chip_updates(&self, out: &mut String, block: &Block) {
        if matches!(self.tracer_mode, TracerMode::Pure) {
            return;
        }

        let mut chip_counts: BTreeMap<u32, u32> = BTreeMap::new();
        let mut increment_chip_count = |pc: u32| match self.chip_idx_for_pc(pc) {
            TraceChipIndex::Chip(chip) => *chip_counts.entry(chip.as_u32()).or_insert(0) += 1,
            TraceChipIndex::NoChip => {}
        };
        for instr_at in &block.instructions {
            increment_chip_count(instr_at.pc);
        }
        if !matches!(block.terminator, Terminator::FallThrough) {
            increment_chip_count(block.terminator_pc);
        }
        if chip_counts.is_empty() {
            return;
        }

        writeln!(out, "    {{").unwrap();
        match self.tracer_mode {
            TracerMode::Pure => unreachable!(),
            TracerMode::Metered => {
                writeln!(
                    out,
                    "        uint32_t* restrict _trace_heights = state->tracer->trace_heights;"
                )
                .unwrap();
                for (chip, count) in &chip_counts {
                    writeln!(out, "        _trace_heights[{chip}] += {count}u;").unwrap();
                }
            }
            TracerMode::MeteredCost => {
                let widths = self.chip_widths.as_ref().unwrap();
                let total: u64 = chip_counts
                    .iter()
                    .map(|(&chip, &count)| widths[chip as usize] * count as u64)
                    .sum();
                if total > 0 {
                    writeln!(out, "        state->tracer->cost += {total}u;").unwrap();
                }
            }
        }
        writeln!(out, "    }}").unwrap();
    }

    /// Emit PC/opname comment and `#line` directive for source attribution.
    fn emit_source_annotation(
        &self,
        out: &mut String,
        pc: u32,
        opname: &str,
        source_loc: Option<&SourceLoc>,
    ) {
        match source_loc {
            Some(loc) if loc.is_valid() => {
                if loc.function.is_empty() {
                    writeln!(out, "    // 0x{pc:08x}  {opname}").unwrap();
                } else {
                    writeln!(out, "    // 0x{pc:08x}  {opname}  @ {}", loc.function).unwrap();
                }
                let escaped = Self::escape_c_string_literal(&loc.file);
                writeln!(out, "#line {} \"{}\"", loc.line, escaped).unwrap();
            }
            _ => {
                writeln!(out, "    // 0x{pc:08x}  {opname}").unwrap();
            }
        }
    }

    // ── Dispatch ─────────────────────────────────────────────────────────

    fn write_dispatch(
        &self,
        blocks: &[Block],
        entry_point: u32,
        text_start: u32,
    ) -> io::Result<()> {
        let name = &self.name;
        let mut src = String::with_capacity(16 * 1024);

        let params = self.param_list();
        let args_from_state = self.fn_args_from_state();

        writeln!(src, "#include \"{name}.h\"").unwrap();
        writeln!(src).unwrap();

        let save = self.save_hot_regs_call();
        // rv_trap — cold fallback for dispatch to non-block PCs.
        writeln!(
            src,
            "static __attribute__((preserve_none, cold)) void rv_trap({params}) {{"
        )
        .unwrap();
        writeln!(src, "    {save}").unwrap();
        writeln!(src, "    state->has_exited = OPENVM_EXEC_TRAPPED;").unwrap();
        writeln!(src, "    state->exit_code = 0;").unwrap();
        writeln!(src, "    return;").unwrap();
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();

        // Dense dispatch table: one entry per 4-byte instruction slot.
        let max_pc = blocks
            .iter()
            .map(|b| b.start_pc)
            .chain(std::iter::once(entry_point))
            .max()
            .unwrap_or(text_start);
        debug_assert!(max_pc >= text_start);
        let table_size = ((max_pc - text_start) / 4 + 1) as usize;

        // Build block-start lookup.
        let block_starts: std::collections::HashMap<u32, u32> =
            blocks.iter().map(|b| (b.start_pc, b.start_pc)).collect();

        writeln!(src, "BlockFn dispatch_table[{table_size}] = {{").unwrap();
        for i in 0..table_size {
            let pc = text_start + (i as u32) * 4;
            if block_starts.contains_key(&pc) {
                writeln!(src, "    block_0x{pc:08x},").unwrap();
            } else {
                writeln!(src, "    rv_trap,").unwrap();
            }
        }
        writeln!(src, "}};").unwrap();
        writeln!(src).unwrap();

        // rv_execute — single entry call; tail calls chain blocks.
        writeln!(
            src,
            "__attribute__((hot, nonnull)) void rv_execute(RvState* restrict state) {{"
        )
        .unwrap();
        writeln!(
            src,
            "    if (state->pc < 0x{text_start:08x}u || state->pc > 0x{max_pc:08x}u) {{"
        )
        .unwrap();
        writeln!(src, "        rv_trap({args_from_state});").unwrap();
        writeln!(src, "        return;").unwrap();
        writeln!(src, "    }}").unwrap();
        writeln!(src, "    uint32_t idx = dispatch_index(state->pc);").unwrap();
        writeln!(src, "    dispatch_table[idx]({args_from_state});").unwrap();
        writeln!(src, "    return;").unwrap();
        writeln!(src, "}}").unwrap();

        let path = self.output_dir.join("dispatch.c");
        fs::write(&path, src)
    }

    // ── Makefile ─────────────────────────────────────────────────────────

    fn write_makefile(&self) -> io::Result<()> {
        let path = self.output_dir.join("Makefile");
        fs::write(&path, include_str!("../../c/Makefile"))
    }

    /// Make arguments to set LIB and (optionally) LTO/DEBUG variables.
    pub fn make_args(&self) -> Vec<String> {
        let lib_ext = if cfg!(target_os = "macos") {
            "dylib"
        } else {
            "so"
        };

        let mut args = vec![format!("LIB=lib{}.{lib_ext}", self.name)];
        if self.enable_lto {
            args.push("LTO=-flto=thin".to_string());
        }
        if self.native_debug_info {
            args.push("DEBUG=-g -fno-omit-frame-pointer".to_string());
        }
        args
    }

    pub fn make_args_with_extensions(
        &self,
        ext_staticlibs: &[PathBuf],
        ext_sources: &[String],
        ext_cflags: &[String],
    ) -> Vec<String> {
        let mut args = self.make_args();
        if !ext_sources.is_empty() {
            args.push(format!("EXT_SRCS={}", ext_sources.join(" ")));
        }
        if !ext_staticlibs.is_empty() {
            // `make` receives `VAR=value` assignments as single argv entries,
            // but `$(EXT_LIBS)` and `$(EXT_CFLAGS)` are later expanded by
            // splitting on spaces. Static library paths, `-I...` paths, and
            // copied extra-source filenames in `$(EXT_SRCS)` therefore must
            // not contain spaces.
            let libs: Vec<String> = ext_staticlibs
                .iter()
                .map(|p| p.display().to_string())
                .collect();
            args.push(format!("EXT_LIBS={}", libs.join(" ")));
        }
        if !ext_cflags.is_empty() {
            args.push(format!("EXT_CFLAGS={}", ext_cflags.join(" ")));
        }
        args
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tracer_modes_select_expected_default_suspend_policy() {
        assert_eq!(
            TracerMode::Pure.default_suspend_policy(),
            SuspendPolicy::InstretLimit
        );
        assert_eq!(
            TracerMode::MeteredCost.default_suspend_policy(),
            SuspendPolicy::InstretLimit
        );
        assert_eq!(
            TracerMode::Metered.default_suspend_policy(),
            SuspendPolicy::Disabled
        );
    }

    #[test]
    fn metered_segment_boundary_selects_segment_block_header() {
        let normal_metered = TracerMode::Metered.block_header_content(SuspendPolicy::Disabled);
        let segment_metered =
            TracerMode::Metered.block_header_content(SuspendPolicy::SegmentBoundary);

        assert!(normal_metered.contains("trace_block(state, pc, block_insn_count);"));
        assert!(!normal_metered.contains("trace_block_with_segment_check"));
        assert!(segment_metered.contains("trace_block_with_segment_check"));
    }

    #[test]
    fn segment_boundary_suspender_uses_block_signal() {
        let suspender = SuspendPolicy::SegmentBoundary.header_content();

        assert!(suspender.contains("uint8_t suspend_signal"));
        assert!(suspender.contains("return suspend_signal;"));
    }
}
