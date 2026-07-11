//! C project generation: header, block files, dispatch, Makefile.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write,
    fs, io,
    path::{Path, PathBuf},
};

use rvr_openvm_ir::*;
use rvr_openvm_lift::{ExtensionRegistry, TraceChipIndex};
use rvr_state::NUM_REGS;

use super::{
    codegen::{emit_terminator, TermCtx},
    context::{validate_chip_index, BlockAbi, EmitContext, EmitMode, InvalidChipIndex},
};
use crate::constants::constants_header;

/// Complete execution behavior baked into a generated RVR artifact.
///
/// The numeric values are part of the dynamic-library ABI through the
/// `rv_execution_kind` export.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RvrExecutionKind {
    Pure = 0,
    PureWithInstretTracking = 1,
    MeteredCost = 2,
    Metered = 3,
    MeteredSegment = 4,
    /// Preflight execution logging program and memory accesses.
    Preflight = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown RVR execution kind {0}")]
pub struct InvalidRvrExecutionKind(pub u32);

impl TryFrom<u32> for RvrExecutionKind {
    type Error = InvalidRvrExecutionKind;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Pure),
            1 => Ok(Self::PureWithInstretTracking),
            2 => Ok(Self::MeteredCost),
            3 => Ok(Self::Metered),
            4 => Ok(Self::MeteredSegment),
            5 => Ok(Self::Preflight),
            value => Err(InvalidRvrExecutionKind(value)),
        }
    }
}

impl RvrExecutionKind {
    pub const fn artifact_suffix(self) -> &'static str {
        match self {
            Self::Pure => "pure",
            Self::PureWithInstretTracking => "pure-with-instret-tracking",
            Self::MeteredCost => "metered-cost",
            Self::Metered => "metered",
            Self::MeteredSegment => "metered-segment",
            Self::Preflight => "preflight",
        }
    }

    fn trace_header_filename(self) -> &'static str {
        match self {
            Self::Pure | Self::PureWithInstretTracking => "openvm_tracer_pure.h",
            Self::MeteredCost => "openvm_tracer_metered_cost.h",
            Self::Metered | Self::MeteredSegment => "openvm_tracer_metered.h",
            Self::Preflight => "openvm_tracer_preflight.h",
        }
    }

    fn trace_header_content(self) -> &'static str {
        match self {
            Self::Pure | Self::PureWithInstretTracking => {
                include_str!("../../c/tracer/openvm_tracer_pure.h")
            }
            Self::MeteredCost => include_str!("../../c/tracer/openvm_tracer_metered_cost.h"),
            Self::Metered | Self::MeteredSegment => {
                include_str!("../../c/tracer/openvm_tracer_metered.h")
            }
            Self::Preflight => include_str!("../../c/tracer/openvm_tracer_preflight.h"),
        }
    }

    fn metered_block_header_content(self) -> Option<&'static str> {
        match self {
            Self::Metered => Some(include_str!("../../c/block/openvm_block_metered.h")),
            Self::MeteredSegment => {
                Some(include_str!("../../c/block/openvm_block_metered_segment.h"))
            }
            Self::Pure | Self::PureWithInstretTracking | Self::MeteredCost | Self::Preflight => None,
        }
    }

    fn state_layout_header(self) -> String {
        let mut out = String::new();
        writeln!(out, "#ifndef OPENVM_STATE_LAYOUT_H").unwrap();
        writeln!(out, "#define OPENVM_STATE_LAYOUT_H").unwrap();
        writeln!(out).unwrap();
        match self {
            Self::Pure | Self::Preflight => {}
            Self::PureWithInstretTracking => {
                writeln!(out, "typedef struct InstretTrackingState {{").unwrap();
                writeln!(out, "  uint64_t retired;").unwrap();
                writeln!(out, "  uint64_t target;").unwrap();
                writeln!(out, "}} InstretTrackingState;").unwrap();
            }
            Self::MeteredCost => {
                writeln!(out, "typedef struct MeteredCostState {{").unwrap();
                writeln!(out, "  uint64_t instret;").unwrap();
                writeln!(out, "  uint64_t cost;").unwrap();
                writeln!(out, "}} MeteredCostState;").unwrap();
            }
            Self::Metered | Self::MeteredSegment => {
                writeln!(out, "typedef uint32_t TraceHeights[RV_NUM_AIRS];").unwrap();
                writeln!(out, "struct PageTouch;").unwrap();
                writeln!(out, "struct SegmentationState;").unwrap();
                writeln!(out, "typedef struct MeteringState {{").unwrap();
                writeln!(out, "  TraceHeights* trace_heights;").unwrap();
                writeln!(out, "  struct PageTouch* mem_page_buf;").unwrap();
                writeln!(out, "  struct PageTouch* pv_page_buf;").unwrap();
                writeln!(out, "  struct PageTouch* deferral_page_buf;").unwrap();
                writeln!(out, "  uint8_t (*on_check)(struct MeteringState*);").unwrap();
                writeln!(out, "  void (*on_memory_flush)(struct MeteringState*);").unwrap();
                writeln!(out, "  struct SegmentationState* seg_state;").unwrap();
                writeln!(out, "  uint32_t mem_page_buf_len;").unwrap();
                writeln!(out, "  uint32_t pv_page_buf_len;").unwrap();
                writeln!(out, "  uint32_t deferral_page_buf_len;").unwrap();
                writeln!(out, "  uint32_t check_counter;").unwrap();
                writeln!(out, "  uint32_t last_mem_page;").unwrap();
                writeln!(
                    out,
                    "  /* Explicit tail padding required by pointer alignment. */"
                )
                .unwrap();
                writeln!(out, "  uint32_t padding;").unwrap();
                writeln!(out, "}} MeteringState;").unwrap();
            }
        }
        writeln!(out).unwrap();
        writeln!(out, "typedef struct RvState {{").unwrap();
        writeln!(out, "  uint64_t regs[{NUM_REGS}];").unwrap();
        writeln!(out, "  uint64_t pc;").unwrap();
        writeln!(out, "  uint8_t status;").unwrap();
        writeln!(out, "  uint8_t exit_code;").unwrap();
        writeln!(out, "  /* Keep the following pointer naturally aligned. */").unwrap();
        writeln!(out, "  uint8_t padding[6];").unwrap();
        writeln!(out, "  uint8_t* memory;").unwrap();
        match self {
            Self::Pure | Self::Preflight => {}
            Self::PureWithInstretTracking => {
                writeln!(out, "  InstretTrackingState mode_state;").unwrap();
            }
            Self::MeteredCost => {
                writeln!(out, "  MeteredCostState mode_state;").unwrap();
            }
            Self::Metered | Self::MeteredSegment => {
                writeln!(out, "  MeteringState mode_state;").unwrap();
            }
        }
        writeln!(out, "}} RvState;").unwrap();
        writeln!(out).unwrap();
        writeln!(out, "#endif /* OPENVM_STATE_LAYOUT_H */").unwrap();
        out
    }

    fn write_execution_kind_marker(self, out: &mut String) {
        writeln!(out, "uint32_t rv_execution_kind(void);").unwrap();
        writeln!(out, "__attribute__((visibility(\"default\"), used))").unwrap();
        writeln!(out, "uint32_t rv_execution_kind(void) {{").unwrap();
        writeln!(out, "  return {}u;", self as u32).unwrap();
        writeln!(out, "}}").unwrap();
    }

    const fn block_abi(self) -> BlockAbi {
        match self {
            Self::Pure | Self::MeteredCost | Self::Preflight => BlockAbi::Plain,
            Self::PureWithInstretTracking => BlockAbi::InstretCountdown,
            Self::Metered | Self::MeteredSegment => BlockAbi::Metered,
        }
    }
}

/// C project generator.
pub struct CProject {
    output_dir: PathBuf,
    name: String,
    execution_kind: RvrExecutionKind,
    hot_regs: HashSet<u8>,
    /// Maximum blocks per partition file.
    pub blocks_per_partition: usize,
    /// Enable thin LTO for the generated C code.
    pub enable_lto: bool,
    /// Main chip index for each PC, used to count rows once per block.
    /// Index i = chip for PC = pc_base + i*4.
    /// `None` in pure mode (no chip metadata requested); must be set in metered modes.
    pub pc_to_chip: Option<Vec<TraceChipIndex>>,
    /// Program PC base (used to compute pc_to_chip index).
    pub pc_base: u64,
    /// Per-AIR widths for MeteredCost precomputation. Indexed by chip index.
    pub chip_widths: Option<Vec<u64>>,
    /// Number of AIRs written into a metered artifact's C ABI.
    pub num_airs: Option<u32>,
    /// Compile with native debug info (`-g -fno-omit-frame-pointer`).
    pub native_debug_info: bool,
    /// R3: emit inline compact records (log-suppressed) for migrated opcodes.
    /// Preflight mode only; see [`Self::inline_records_enabled`].
    pub inline_records: bool,
    /// R4: airs whose records the generated C writes arena-native — full
    /// records at final arena positions, field offsets baked as literals
    /// from the geometry's layout table. Airs absent here keep the compact
    /// wire. Populated by the host compile pipeline from the assembler
    /// registry; empty means pure R3 emission.
    pub arena_native_airs: std::collections::BTreeMap<u32, crate::ArenaNativeGeometry>,
}

impl CProject {
    pub fn new(output_dir: &Path, name: &str, execution_kind: RvrExecutionKind) -> Self {
        // Hot registers in priority order.
        // Limited by platform's preserve_none register capacity minus 1 (state ptr).
        let hot_regs = Self::hot_regs_for_kind(execution_kind);

        Self {
            output_dir: output_dir.to_path_buf(),
            name: name.to_string(),
            execution_kind,
            hot_regs,
            blocks_per_partition: 512,
            enable_lto: true,
            pc_to_chip: None,
            pc_base: 0,
            chip_widths: None,
            num_airs: None,
            native_debug_info: false,
            inline_records: false,
            arena_native_airs: std::collections::BTreeMap::new(),
        }
    }

    /// Register priority order.
    /// x0 (zero) is excluded since it's always 0.
    const REG_PRIORITY: [u8; NUM_REGS - 1] = [
        1, 2, // ra, sp
        10, 11, 12, 13, 14, 15, 16, 17, // a0-a7
        5, 6, 7, // t0-t2
        28, 29, 30, 31, // t3-t6
        8, 9, // s0-s1
        18, 19, 20, 21, 22, 23, 24, 25, 26, 27, // s2-s11
        3, 4, // gp, tp
    ];

    /// Maximum non-state arguments supported by the platform's `preserve_none` ABI.
    const fn max_non_state_args() -> usize {
        #[cfg(target_arch = "aarch64")]
        return 23;
        #[cfg(not(target_arch = "aarch64"))]
        return 10;
    }

    /// Reserve argument registers for values carried through the block ABI.
    fn hot_regs_for_kind(kind: RvrExecutionKind) -> HashSet<u8> {
        if kind == RvrExecutionKind::Preflight {
            return HashSet::new();
        }
        let count = Self::max_non_state_args() - kind.block_abi().extra_args();
        Self::REG_PRIORITY[..count].iter().copied().collect()
    }

    pub const fn execution_kind(&self) -> RvrExecutionKind {
        self.execution_kind
    }

    fn block_abi(&self) -> BlockAbi {
        self.execution_kind.block_abi()
    }

    fn validate_block_abi(&self) -> io::Result<()> {
        let used = self.hot_regs.len() + self.block_abi().extra_args();
        if used > Self::max_non_state_args() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "generated block ABI needs {used} non-state arguments, but preserve_none supports {} on this target",
                    Self::max_non_state_args()
                ),
            ));
        }
        Ok(())
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

    fn append_hot_reg_args_from_state(&self, out: &mut String) {
        for &(idx, _) in &self.sorted_hot_regs() {
            write!(out, ", state->regs[{idx}]").unwrap();
        }
    }

    fn append_hot_reg_args_from_params(&self, out: &mut String) {
        for &(_, name) in &self.sorted_hot_regs() {
            write!(out, ", {name}").unwrap();
        }
    }

    fn append_block_abi_args_from_state(&self, out: &mut String) {
        match self.block_abi() {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => {
                out.push_str(", state->mode_state.target - state->mode_state.retired");
            }
            BlockAbi::Metered => {
                out.push_str(", state->mode_state.check_counter, state->mode_state.trace_heights");
            }
        }
    }

    fn append_block_abi_args_from_params(&self, out: &mut String) {
        match self.block_abi() {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => out.push_str(", instret_remaining"),
            BlockAbi::Metered => out.push_str(", check_counter, trace_heights"),
        }
    }

    /// Single-line C function parameter list for state + hot guest regs.
    fn hot_regs_param_list(&self) -> String {
        let mut s = "RvState* restrict state".to_string();
        for &(_, name) in &self.sorted_hot_regs() {
            write!(s, ", uint64_t {name}").unwrap();
        }
        s
    }

    /// C function parameter list entries.
    fn param_list_items(&self, include_names: bool, unused_trace_heights: bool) -> Vec<String> {
        let mut params = vec![if include_names {
            "RvState* restrict state".to_string()
        } else {
            "RvState* restrict".to_string()
        }];
        for &(_, name) in &self.sorted_hot_regs() {
            params.push(if include_names {
                format!("uint64_t {name}")
            } else {
                "uint64_t".to_string()
            });
        }
        match self.block_abi() {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => params.push(if include_names {
                "uint64_t instret_remaining".to_string()
            } else {
                "uint64_t".to_string()
            }),
            BlockAbi::Metered => {
                params.push(if include_names {
                    "uint32_t check_counter".to_string()
                } else {
                    "uint32_t".to_string()
                });
                params.push(if include_names {
                    let attribute = if unused_trace_heights {
                        " [[maybe_unused]]"
                    } else {
                        ""
                    };
                    format!("TraceHeights* restrict trace_heights{attribute}")
                } else {
                    "TraceHeights* restrict".to_string()
                });
            }
        }
        params
    }

    fn function_signature(&self, prefix: &str, name: &str, unused_trace_heights: bool) -> String {
        let params = self.param_list_items(true, unused_trace_heights);
        let mut out = format!("{prefix} {name}(\n");
        for (idx, param) in params.iter().enumerate() {
            let suffix = if idx + 1 == params.len() { "" } else { "," };
            writeln!(out, "    {param}{suffix}").unwrap();
        }
        out.push(')');
        out
    }

    fn block_signature(&self, prefix: &str, name: &str) -> String {
        self.function_signature(prefix, name, false)
    }

    fn trap_signature(&self) -> String {
        self.function_signature("__attribute__((preserve_none, cold)) void", "rv_trap", true)
    }

    fn emit_trap_declaration(&self, out: &mut String) {
        let trap_signature = self.trap_signature();
        writeln!(
            out,
            "// Used when a computed jump or dispatch slot does not point to a real block."
        )
        .unwrap();
        writeln!(
            out,
            "// It takes the same arguments as a block so those register values can still be saved."
        )
        .unwrap();
        writeln!(out, "{trap_signature};").unwrap();
    }
    /// C argument list extracting hot regs from state:
    /// "state, state->regs[1], state->regs[2]".
    fn fn_args_from_state(&self) -> String {
        let mut s = "state".to_string();
        self.append_hot_reg_args_from_state(&mut s);
        self.append_block_abi_args_from_state(&mut s);
        s
    }

    /// C argument list forwarding the current block ABI parameters.
    fn fn_args_from_params(&self) -> String {
        let mut s = "state".to_string();
        self.append_hot_reg_args_from_params(&mut s);
        self.append_block_abi_args_from_params(&mut s);
        s
    }

    /// C typedef parameter types: "RvState*, uint32_t, uint32_t".
    fn typedef_params(&self) -> String {
        self.param_list_items(false, false).join(", ")
    }

    /// Call to save hot registers back to state before returning.
    fn save_hot_regs_call(&self) -> String {
        let mut s = "rv_save_hot_regs(state".to_string();
        self.append_hot_reg_args_from_params(&mut s);
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

    fn block_end_pc(&self, block: &Block) -> u64 {
        block.terminator_pc.saturating_add(4)
    }

    fn dispatch_max_pc(blocks: &[Block], entry_point: u64, text_start: u64) -> u64 {
        blocks
            .iter()
            .map(|b| b.start_pc)
            .chain(std::iter::once(entry_point))
            .max()
            .unwrap_or(text_start)
    }

    fn dispatch_table_size(text_start: u64, text_end: u64) -> usize {
        debug_assert!(text_end >= text_start);
        ((text_end - text_start) / 4 + 1) as usize
    }

    fn emit_mode_for_block(&self, block: &Block) -> EmitMode {
        match self.execution_kind {
            RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => EmitMode::Metered {
                trace_memory_pages: block_accesses_memory(block),
            },
            RvrExecutionKind::MeteredCost => EmitMode::MeteredCost,
            RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => EmitMode::Direct,
            RvrExecutionKind::Preflight => EmitMode::ValueTrace,
        }
    }

    fn emit_context_scope(out: &mut String, ctx: &mut EmitContext) {
        let body = ctx.take_buf();
        if body.is_empty() {
            return;
        }
        out.push_str("    {\n");
        out.push_str(&body);
        out.push_str("    }\n");
    }

    /// R3: whether the preflight codegen emits inline compact records for
    /// migrated opcodes (base-ALU ADD/SUB), suppressing their memory-log
    /// entries and writing the record into the chip's record buffer instead.
    /// Set by the compile pipeline (see `crates/vm` rvr `compile.rs`, which
    /// also derives the matching host-side skip/adopt metadata) and only
    /// active in preflight mode, where `pc_to_chip` is set.
    fn inline_records_enabled(&self) -> bool {
        self.tracer_mode == TracerMode::Preflight
            && self.pc_to_chip.is_some()
            && self.inline_records
    }

    /// Look up the chip index for a given PC. Must only be called in metered
    /// modes; panics if `pc_to_chip` is unset.
    fn chip_idx_for_pc(&self, pc: u64) -> TraceChipIndex {
        let mapping = self
            .pc_to_chip
            .as_ref()
            .expect("pc_to_chip must be set for metered/preflight rvr codegen");
        let Some(offset) = pc.checked_sub(self.pc_base) else {
            return TraceChipIndex::NoChip;
        };
        mapping
            .get((offset / 4) as usize)
            .copied()
            .unwrap_or(TraceChipIndex::NoChip)
    }

    /// Write all C project files.
    pub fn write_all(
        &self,
        blocks: &[Block],
        entry_point: u64,
        text_start: u64,
        extensions: &ExtensionRegistry,
    ) -> io::Result<()> {
        if !matches!(
            self.execution_kind,
            RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking
        ) && self.num_airs.is_none()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "metered RVR code generation requires the AIR count",
            ));
        }
        self.validate_block_abi()?;
        let text_end = Self::dispatch_max_pc(blocks, entry_point, text_start);
        let table_size = Self::dispatch_table_size(text_start, text_end);

        self.write_constants(
            text_start,
            text_end,
            table_size,
            extensions.max_main_memory_pages_per_instruction(),
        )?;
        self.write_support_files()?;
        self.write_extension_files(extensions)?;
        let ext_headers = extensions.c_headers();
        self.write_header(blocks, &ext_headers)?;
        self.write_block_files(blocks)?;
        self.write_dispatch(blocks, entry_point, text_start)?;
        self.write_makefile()?;
        Ok(())
    }

    // ── Generated constants header ──────────────────────────────────────

    fn write_constants(
        &self,
        text_start: u64,
        text_end: u64,
        dispatch_table_size: usize,
        max_mem_pages_per_insn: usize,
    ) -> io::Result<()> {
        let h = constants_header(
            text_start,
            text_end,
            dispatch_table_size,
            self.num_airs,
            max_mem_pages_per_insn,
        );
        let path = self.output_dir.join("openvm_constants.h");
        fs::write(&path, h)
    }

    // ── Support files (trace header, state header, IO) ──────────────────

    fn write_support_files(&self) -> io::Result<()> {
        fs::write(
            self.output_dir.join("openvm_util.h"),
            include_str!("../../c/openvm_util.h"),
        )?;

        // RvState definition specialized to this execution kind.
        fs::write(
            self.output_dir.join("openvm_state_layout.h"),
            self.execution_kind.state_layout_header(),
        )?;
        let state_path = self.output_dir.join("openvm_state.h");
        fs::write(&state_path, include_str!("../../c/openvm_state.h"))?;

        // Mode-specific tracing helpers include the generated state definition.
        let trace_path = self
            .output_dir
            .join(self.execution_kind.trace_header_filename());
        fs::write(&trace_path, self.execution_kind.trace_header_content())?;

        // Metered checkpoint helpers stay in a specialized header. Pure and
        // metered-cost artifacts have no generic block-checking layer.
        if let Some(content) = self.execution_kind.metered_block_header_content() {
            fs::write(self.output_dir.join("openvm_block.h"), content)?;
        }

        // Memory-bounds checks are header-selected so hot helpers can inline.
        let bounds_h_path = self.output_dir.join("openvm_check_mem_bounds.h");
        #[cfg(not(feature = "unprotected"))]
        let bounds_h_content = include_str!("../../c/openvm_check_mem_bounds.h");
        #[cfg(feature = "unprotected")]
        let bounds_h_content = include_str!("../../c/openvm_check_mem_bounds_unprotected.h");
        fs::write(&bounds_h_path, bounds_h_content)?;

        // Protected mode keeps only the cold abort path out-of-line.
        let bounds_c_path = self.output_dir.join("openvm_check_mem_bounds.c");
        #[cfg(not(feature = "unprotected"))]
        fs::write(
            &bounds_c_path,
            include_str!("../../c/openvm_check_mem_bounds.c"),
        )?;
        #[cfg(feature = "unprotected")]
        let _ = fs::remove_file(&bounds_c_path);

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
            self.execution_kind.trace_header_filename()
        )
        .unwrap();
        writeln!(openvm_h, "#include \"openvm_io.h\"").unwrap();
        fs::write(self.output_dir.join("openvm.h"), openvm_h)?;

        Ok(())
    }

    // ── Extension files ─────────────────────────────────────────────────

    fn write_extension_files(&self, extensions: &ExtensionRegistry) -> io::Result<()> {
        let mut created_dirs = HashSet::new();
        self.write_embedded_files(extensions.c_headers(), &mut created_dirs)?;
        self.write_embedded_files(extensions.c_sources(), &mut created_dirs)?;
        self.write_embedded_files(extensions.vendored_c_sources(), &mut created_dirs)?;
        self.write_embedded_files(extensions.extra_c_include_files(), &mut created_dirs)?;

        if extensions.uses_memory_wrappers() {
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
            self.output_dir.join("rvr_ext_wrappers.h"),
            include_str!("../../c/rvr_ext_wrappers.h"),
        )?;
        fs::write(
            self.output_dir.join("rvr_ext_wrappers.c"),
            include_str!("../../c/rvr_ext_wrappers.c"),
        )?;
        Ok(())
    }

    // ── Main header ──────────────────────────────────────────────────────

    fn write_header(&self, blocks: &[Block], ext_headers: &[(&str, &str)]) -> io::Result<()> {
        let name = &self.name;
        let trace_header = self.execution_kind.trace_header_filename();
        let mut h = String::with_capacity(4096);

        // Mode-specific tracing helpers include openvm_state.h internally.
        writeln!(h, "#include \"{trace_header}\"").unwrap();
        writeln!(h).unwrap();

        // Block function type and dispatch table for indirect-jump tail calls.
        let typedef_params = self.typedef_params();
        writeln!(
            h,
            "typedef __attribute__((preserve_none)) void (*BlockFn)({typedef_params});"
        )
        .unwrap();
        self.emit_trap_declaration(&mut h);
        if self.execution_kind == RvrExecutionKind::PureWithInstretTracking {
            let signature = self.block_signature(
                "__attribute__((preserve_none, noinline)) void",
                "rv_suspend",
            );
            writeln!(h, "{signature};").unwrap();
        }
        // Keep this table private to the library so the linker can address it directly.
        writeln!(
            h,
            "extern __attribute__((visibility(\"hidden\"))) BlockFn const dispatch_table[RV_DISPATCH_TABLE_SIZE];"
        )
        .unwrap();
        writeln!(h).unwrap();
        let trap_signature =
            self.block_signature("__attribute__((preserve_none, cold)) void", "rv_trap");
        writeln!(h, "{trap_signature};").unwrap();
        writeln!(h).unwrap();

        // Block function declarations.
        let hot_params = self.hot_regs_param_list();
        for block in blocks {
            let signature = self.block_signature(
                "__attribute__((preserve_none)) void",
                &format!("block_0x{:08x}", block.start_pc),
            );
            writeln!(h, "{signature};").unwrap();
        }
        writeln!(h).unwrap();

        writeln!(
            h,
            "static __attribute__((always_inline)) inline void rv_save_hot_regs({hot_params}) {{"
        )
        .unwrap();
        for &(idx, name) in &self.sorted_hot_regs() {
            writeln!(h, "    state->regs[{idx}] = {name};").unwrap();
        }
        writeln!(h, "}}").unwrap();
        writeln!(h).unwrap();

        // Runtime and extension headers.
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
        let valid_blocks: HashSet<u64> = blocks.iter().map(|b| b.start_pc).collect();

        for part_idx in 0..num_partitions {
            let start = part_idx * self.blocks_per_partition;
            let end = (start + self.blocks_per_partition).min(blocks.len());
            let partition = &blocks[start..end];

            let first_pc = partition[0].start_pc;
            let mut src = String::with_capacity(64 * 1024);
            writeln!(src, "#include \"{name}.h\"").unwrap();
            if self.execution_kind.metered_block_header_content().is_some() {
                writeln!(src, "#include \"openvm_block.h\"").unwrap();
            }
            writeln!(src).unwrap();

            for block in partition {
                self.emit_block_checkpoint_function(&mut src, block);
                self.emit_block_function(&mut src, block, &valid_blocks)
                    .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
            }

            let path = self.output_dir.join(format!("{name}_0x{first_pc:08x}.c"));
            fs::write(&path, src)?;
        }

        Ok(())
    }

    fn emit_block_function(
        &self,
        out: &mut String,
        block: &Block,
        valid_blocks: &HashSet<u64>,
    ) -> Result<(), InvalidChipIndex> {
        let pc = block.start_pc;
        let end_pc = self.block_end_pc(block);
        let insn_count = block.insn_count();
        let mode = self.emit_mode_for_block(block);

        writeln!(
            out,
            "// Block: 0x{pc:08x}-0x{:08x} ({insn_count} instrs)",
            end_pc.saturating_sub(1)
        )
        .unwrap();
        // Body instructions (each in its own scope to avoid variable collisions).
        let chip_widths = match mode {
            EmitMode::MeteredCost => self.chip_widths.as_deref(),
            _ => None,
        };
        let mut ctx = EmitContext::new(
            self.hot_regs.clone(),
            mode,
            self.block_abi(),
            chip_widths,
            self.num_airs,
        );
        let mut body = String::new();
        let inline_records = self.inline_records_enabled();
        ctx.set_inline_records(inline_records);
        if inline_records && !self.arena_native_airs.is_empty() {
            ctx.set_arena_native_airs(self.arena_native_airs.clone());
        }

        if matches!(
            mode,
            EmitMode::Metered {
                trace_memory_pages: true
            }
        ) {
            writeln!(
                body,
                "    TraceMemory trace_memory = trace_memory_setup(&state->mode_state);"
            )
            .unwrap();
        }

        self.emit_block_boundary(&mut body, block);
        self.emit_per_block_chip_updates(&mut body, block)?;

        for instr_at in &block.instructions {
            self.emit_source_annotation(
                &mut body,
                instr_at.pc,
                instr_at.instr.opname(),
                instr_at.source_loc.as_ref(),
            );
            if inline_records {
                let chip_idx = match self.chip_idx_for_pc(instr_at.pc) {
                    TraceChipIndex::Chip(air) => air.as_u32(),
                    TraceChipIndex::NoChip => u32::MAX,
                };
                ctx.set_current_instr(chip_idx, instr_at.pc);
            }
            ctx.trace_pc(instr_at.pc);
            instr_at.instr.emit_c(&mut ctx);
            Self::emit_context_scope(&mut body, &mut ctx);
            body.push('\n');
        }

        ctx.flush_page_locals();
        Self::emit_context_scope(&mut body, &mut ctx);

        if !matches!(block.terminator, Terminator::FallThrough) {
            self.emit_source_annotation(
                &mut body,
                block.terminator_pc,
                block.terminator.opname(),
                block.terminator_source_loc.as_ref(),
            );
            ctx.trace_pc(block.terminator_pc);
        }
        if inline_records {
            let chip_idx = match self.chip_idx_for_pc(block.terminator_pc) {
                TraceChipIndex::Chip(air) => air.as_u32(),
                TraceChipIndex::NoChip => u32::MAX,
            };
            ctx.set_current_instr(chip_idx, block.terminator_pc);
        }
        let tc = TermCtx { valid_blocks };
        emit_terminator(&mut ctx, &block.terminator, block.terminator_pc, &tc);
        Self::emit_context_scope(&mut body, &mut ctx);

        if let Some(error) = ctx.invalid_chip_index() {
            return Err(error);
        }

        let self_tail_call = format!("return block_0x{pc:08x}(");
        let has_direct_self_tail_call = body.contains(&self_tail_call);
        if has_direct_self_tail_call {
            writeln!(
                out,
                "// This guest back-edge is an intentional musttail self-call."
            )
            .unwrap();
            writeln!(out, "#pragma clang diagnostic push").unwrap();
            writeln!(
                out,
                "#pragma clang diagnostic ignored \"-Winfinite-recursion\""
            )
            .unwrap();
        }
        let signature = self.block_signature(
            "__attribute__((preserve_none)) void",
            &format!("block_0x{pc:08x}"),
        );
        writeln!(out, "{signature} {{").unwrap();
        if ctx.uses_raw_memory() {
            writeln!(out, "    uint8_t* memory = state->memory;").unwrap();
        }
        out.push_str(&body);

        writeln!(out, "}}").unwrap();
        if has_direct_self_tail_call {
            writeln!(out, "#pragma clang diagnostic pop").unwrap();
        }
        writeln!(out).unwrap();
        Ok(())
    }

    fn emit_block_checkpoint_function(&self, out: &mut String, block: &Block) {
        match self.execution_kind {
            RvrExecutionKind::PureWithInstretTracking => return,
            RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {}
            RvrExecutionKind::Pure | RvrExecutionKind::MeteredCost => return,
        }

        let pc = block.start_pc;
        let args = self.fn_args_from_params();

        let signature = self.block_signature(
            "static __attribute__((preserve_none, cold, noinline)) void",
            &format!("block_0x{pc:08x}_checkpoint"),
        );
        writeln!(out, "{signature} {{").unwrap();
        match self.execution_kind {
            RvrExecutionKind::MeteredSegment => {
                self.emit_segment_checkpoint(out, pc);
            }
            RvrExecutionKind::Metered => {
                writeln!(
                    out,
                    "    check_counter = metered_checkpoint(state, check_counter);"
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        writeln!(
            out,
            "    [[clang::musttail]] return block_0x{pc:08x}({args});"
        )
        .unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_instret_suspend_function(&self, out: &mut String) {
        debug_assert_eq!(
            self.execution_kind,
            RvrExecutionKind::PureWithInstretTracking
        );
        let signature = self.block_signature(
            "__attribute__((preserve_none, noinline)) void",
            "rv_suspend",
        );
        writeln!(out, "{signature} {{").unwrap();
        let save = self.save_hot_regs_call();
        writeln!(out, "    {save}").unwrap();
        writeln!(
            out,
            "    state->mode_state.retired = state->mode_state.target - instret_remaining;"
        )
        .unwrap();
        writeln!(out, "    rv_set_status(state, OPENVM_EXEC_SUSPENDED, 0);").unwrap();
        writeln!(out, "    return;").unwrap();
        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_segment_checkpoint(&self, out: &mut String, pc: u64) {
        writeln!(
            out,
            "    MeteredSegmentCheckpointResult checkpoint = metered_segment_checkpoint(state, check_counter);"
        )
        .unwrap();
        writeln!(out, "    check_counter = checkpoint.check_counter;").unwrap();
        writeln!(out, "    if (unlikely(checkpoint.suspend_signal != 0)) {{").unwrap();
        self.emit_suspend_return(out, pc);
        writeln!(out, "    }}").unwrap();
    }

    fn emit_block_boundary(&self, out: &mut String, block: &Block) {
        let pc = block.start_pc;
        let insn_count = block.insn_count();

        match self.execution_kind {
            RvrExecutionKind::Pure => {}
            RvrExecutionKind::PureWithInstretTracking => {
                let args = self.fn_args_from_params();
                writeln!(
                    out,
                    "    if (unlikely(instret_remaining < {insn_count}u)) {{"
                )
                .unwrap();
                writeln!(out, "        state->pc = 0x{pc:08x}ull;").unwrap();
                writeln!(
                    out,
                    "        [[clang::musttail]] return rv_suspend({args});"
                )
                .unwrap();
                writeln!(out, "    }}").unwrap();
                writeln!(out, "    instret_remaining -= {insn_count}u;").unwrap();
            }
            RvrExecutionKind::MeteredCost => {
                writeln!(out, "    state->mode_state.instret += {insn_count}u;").unwrap();
            }
            RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {
                self.emit_metered_counter_check(out, pc, insn_count);
                writeln!(out, "    check_counter -= {insn_count}u;").unwrap();
            }
        }
    }

    fn emit_metered_counter_check(&self, out: &mut String, pc: u64, insn_count: u32) {
        let args = self.fn_args_from_params();
        writeln!(out, "    if (unlikely(check_counter < {insn_count}u)) {{").unwrap();
        writeln!(
            out,
            "        [[clang::musttail]] return block_0x{pc:08x}_checkpoint({args});"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
    }

    fn emit_suspend_return(&self, out: &mut String, pc: u64) {
        let save = self.save_hot_regs_call();
        writeln!(out, "        {save}").unwrap();
        writeln!(
            out,
            "        rv_set_status_at(state, 0x{pc:08x}ull, OPENVM_EXEC_SUSPENDED, 0);"
        )
        .unwrap();
        writeln!(out, "        return;").unwrap();
    }

    /// Sum the chip contributions for every instruction in `block` (body
    /// instructions plus the real terminator instruction, if any) and emit a
    /// batched metering update instead of per-instruction `trace_chip` calls.
    ///
    /// Mode-dependent emission:
    ///   - Pure: nothing (chip tracking is a no-op).
    ///   - Metered: `trace_heights[idx] += count;` per distinct chip, using the trace-heights
    ///     pointer carried through the block ABI.
    ///   - MeteredCost: `state->mode_state.cost += <constant>;` where the constant is
    ///     `sum(width[chip]
    ///     * count)` precomputed at emit time from `self.chip_widths`.
    ///   - Preflight: `trace_chip(state, idx, count);` per distinct chip.
    fn emit_per_block_chip_updates(
        &self,
        out: &mut String,
        block: &Block,
    ) -> Result<(), InvalidChipIndex> {
        if matches!(
            self.execution_kind,
            RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking
        ) {
            return Ok(());
        }

        let num_airs = self
            .num_airs
            .expect("metered code generation requires the AIR count");
        let mut chip_counts: BTreeMap<u32, u32> = BTreeMap::new();
        let add_rows = |chip_counts: &mut BTreeMap<u32, u32>, chip_idx, count| {
            validate_chip_index(chip_idx, num_airs)?;
            let current = chip_counts.entry(chip_idx).or_default();
            *current = current
                .checked_add(count)
                .expect("per-block trace-height increment overflow");
            Ok::<(), InvalidChipIndex>(())
        };
        let add_base_row = |chip_counts: &mut BTreeMap<u32, u32>, pc: u64| {
            if let TraceChipIndex::Chip(chip) = self.chip_idx_for_pc(pc) {
                add_rows(chip_counts, chip.as_u32(), 1)?;
            }
            Ok::<(), InvalidChipIndex>(())
        };
        let add_instruction_rows = |chip_counts: &mut BTreeMap<u32, u32>, instr: &dyn ExtInstr| {
            for rows in instr.fixed_trace_rows() {
                add_rows(chip_counts, rows.chip_idx, rows.count)?;
            }
            Ok::<(), InvalidChipIndex>(())
        };
        for instr_at in &block.instructions {
            add_base_row(&mut chip_counts, instr_at.pc)?;
            add_instruction_rows(&mut chip_counts, instr_at.instr.as_ref())?;
        }
        if !matches!(block.terminator, Terminator::FallThrough) {
            add_base_row(&mut chip_counts, block.terminator_pc)?;
            if let Terminator::Instruction { node, .. } = &block.terminator {
                add_instruction_rows(&mut chip_counts, node.as_ref())?;
            }
        }
        if chip_counts.is_empty() {
            return Ok(());
        }

        writeln!(out, "    {{").unwrap();
        match self.execution_kind {
            RvrExecutionKind::Pure | RvrExecutionKind::PureWithInstretTracking => unreachable!(),
            RvrExecutionKind::Metered | RvrExecutionKind::MeteredSegment => {
                for (chip, count) in &chip_counts {
                    writeln!(out, "        (*trace_heights)[{chip}] += {count}u;").unwrap();
                }
            }
            RvrExecutionKind::MeteredCost => {
                let widths = self.chip_widths.as_ref().unwrap();
                let total = chip_counts.iter().fold(0u64, |total, (&chip, &count)| {
                    let contribution = widths[chip as usize]
                        .checked_mul(u64::from(count))
                        .expect("per-block metered cost contribution overflow");
                    total
                        .checked_add(contribution)
                        .expect("per-block metered cost total overflow")
                });
                if total > 0 {
                    writeln!(out, "        state->mode_state.cost += {total}ull;").unwrap();
                }
            }
            RvrExecutionKind::Preflight => {
                for (chip, count) in &chip_counts {
                    writeln!(out, "        trace_chip(state, {chip}u, {count}u);").unwrap();
                }
            }
        }
        writeln!(out, "    }}").unwrap();
        Ok(())
    }

    /// Emit PC/opname comment and `#line` directive for source attribution.
    fn emit_source_annotation(
        &self,
        out: &mut String,
        pc: u64,
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
        entry_point: u64,
        text_start: u64,
    ) -> io::Result<()> {
        let name = &self.name;
        let mut src = String::with_capacity(16 * 1024);

        let args_from_state = self.fn_args_from_state();

        writeln!(src, "#include \"{name}.h\"").unwrap();
        writeln!(src).unwrap();

        if self.execution_kind == RvrExecutionKind::PureWithInstretTracking {
            self.emit_instret_suspend_function(&mut src);
        }

        let save = self.save_hot_regs_call();
        // rv_trap — cold fallback for dispatch to non-block PCs.
        let trap_signature = self.trap_signature();
        writeln!(src, "{trap_signature} {{").unwrap();
        writeln!(src, "    {save}").unwrap();
        match self.block_abi() {
            BlockAbi::Plain => {}
            BlockAbi::InstretCountdown => {
                writeln!(
                    src,
                    "    state->mode_state.retired = state->mode_state.target - instret_remaining;"
                )
                .unwrap();
            }
            BlockAbi::Metered => {
                writeln!(src, "    state->mode_state.check_counter = check_counter;").unwrap();
            }
        }
        writeln!(src, "    rv_set_status(state, OPENVM_EXEC_TRAPPED, 0);").unwrap();
        writeln!(src, "    return;").unwrap();
        writeln!(src, "}}").unwrap();
        writeln!(src).unwrap();

        // Dense dispatch table: one entry per 4-byte instruction slot.
        let max_pc = Self::dispatch_max_pc(blocks, entry_point, text_start);
        let table_size = Self::dispatch_table_size(text_start, max_pc);

        // Build block-start lookup.
        let block_starts: std::collections::HashMap<u64, u64> =
            blocks.iter().map(|b| (b.start_pc, b.start_pc)).collect();

        writeln!(
            src,
            "BlockFn const dispatch_table[RV_DISPATCH_TABLE_SIZE] = {{"
        )
        .unwrap();
        for i in 0..table_size {
            let pc = text_start + (i as u64) * 4;
            if block_starts.contains_key(&pc) {
                writeln!(src, "    block_0x{pc:08x},").unwrap();
            } else {
                writeln!(src, "    rv_trap,").unwrap();
            }
        }
        writeln!(src, "}};").unwrap();
        writeln!(src).unwrap();

        // Let loaders reject an artifact generated for a different execution kind.
        self.execution_kind.write_execution_kind_marker(&mut src);
        writeln!(src).unwrap();
        if self.num_airs.is_some() {
            writeln!(src, "uint32_t rv_num_airs(void);").unwrap();
            writeln!(src, "__attribute__((visibility(\"default\"), used))").unwrap();
            writeln!(src, "uint32_t rv_num_airs(void) {{").unwrap();
            writeln!(src, "    return RV_NUM_AIRS;").unwrap();
            writeln!(src, "}}").unwrap();
            writeln!(src).unwrap();
        }
        if self.execution_kind == RvrExecutionKind::MeteredCost {
            let widths = self
                .chip_widths
                .as_ref()
                .expect("metered-cost artifact requires chip widths");
            writeln!(
                src,
                "static constexpr uint64_t RV_CHIP_WIDTHS[RV_NUM_AIRS] = {{"
            )
            .unwrap();
            for width in widths {
                writeln!(src, "    {width}ull,").unwrap();
            }
            writeln!(src, "}};").unwrap();
            writeln!(src, "const uint64_t* rv_chip_widths(void);").unwrap();
            writeln!(src, "__attribute__((visibility(\"default\"), used))").unwrap();
            writeln!(src, "const uint64_t* rv_chip_widths(void) {{").unwrap();
            writeln!(src, "    return RV_CHIP_WIDTHS;").unwrap();
            writeln!(src, "}}").unwrap();
            writeln!(src).unwrap();
        }

        // Execution entry point — single entry call; tail calls chain blocks.
        writeln!(
            src,
            "__attribute__((hot, nonnull)) void rv_execute(RvState* restrict state) {{"
        )
        .unwrap();
        writeln!(
            src,
            "    if (unlikely(!rv_pc_is_dispatchable(state->pc))) {{"
        )
        .unwrap();
        writeln!(src, "        rv_trap({args_from_state});").unwrap();
        writeln!(src, "        return;").unwrap();
        writeln!(src, "    }}").unwrap();
        writeln!(src, "    uint64_t idx = rv_dispatch_index(state->pc);").unwrap();
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
        vendor_sources: &[String],
        ext_cflags: &[String],
    ) -> Vec<String> {
        let mut args = self.make_args();
        if !ext_sources.is_empty() {
            args.push(format!("EXT_SRCS={}", ext_sources.join(" ")));
        }
        if !vendor_sources.is_empty() {
            args.push(format!("VENDOR_SRCS={}", vendor_sources.join(" ")));
        }
        if !ext_staticlibs.is_empty() {
            // Make expands these space-separated paths, so extension filenames
            // cannot contain spaces.
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

fn instr_accesses_memory(instr: &dyn ExtInstr) -> bool {
    instr.accesses_memory()
}

fn terminator_accesses_memory(terminator: &Terminator) -> bool {
    match terminator {
        Terminator::Instruction { node, .. } => node.accesses_memory(),
        _ => false,
    }
}

fn block_accesses_memory(block: &Block) -> bool {
    block
        .instructions
        .iter()
        .any(|instr_at| instr_accesses_memory(instr_at.instr.as_ref()))
        || terminator_accesses_memory(&block.terminator)
}

#[cfg(test)]
mod tests {
    use std::{collections::HashSet, path::Path};

    use rvr_openvm_ir::{Block, CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, Terminator};

    use super::{CProject, RvrExecutionKind};

    #[test]
    fn metered_state_layout_includes_memory_flush_callback() {
        let header = RvrExecutionKind::Metered.state_layout_header();
        assert!(header.contains("void (*on_memory_flush)(struct MeteringState*);"));
    }

    fn single_instruction_block() -> Block {
        Block {
            start_pc: 0x100,
            end_pc: 0x104,
            instructions: Vec::new(),
            terminator: Terminator::Exit { code: 0 },
            terminator_pc: 0x100,
            terminator_source_loc: None,
        }
    }

    #[derive(Clone, Debug)]
    struct InvalidTraceChipInstr;

    impl ExtInstr for InvalidTraceChipInstr {
        fn emit_c(&self, ctx: &mut dyn ExtEmitCtx) {
            ctx.trace_chip(1, "1u");
        }

        fn cfg_effect(&self) -> CfgEffect {
            CfgEffect::None
        }

        fn clone_box(&self) -> Box<dyn ExtInstr> {
            Box::new(self.clone())
        }
    }

    #[test]
    fn unlimited_pure_has_plain_block_abi_and_no_boundary_code() {
        let project = CProject::new(Path::new("unused"), "test", RvrExecutionKind::Pure);
        let mut boundary = String::new();
        project.emit_block_boundary(&mut boundary, &single_instruction_block());

        assert!(boundary.is_empty());
        assert!(!project.typedef_params().contains("instret"));

        #[cfg(target_arch = "aarch64")]
        assert_eq!(project.hot_regs.len(), 23);
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(project.hot_regs.len(), 10);
    }

    #[test]
    fn tracked_pure_carries_countdown_and_uses_shared_suspend_path() {
        let project = CProject::new(
            Path::new("unused"),
            "test",
            RvrExecutionKind::PureWithInstretTracking,
        );
        let block = single_instruction_block();
        let mut boundary = String::new();
        project.emit_block_boundary(&mut boundary, &block);
        let mut checkpoint = String::new();
        project.emit_block_checkpoint_function(&mut checkpoint, &block);
        let mut suspend = String::new();
        project.emit_instret_suspend_function(&mut suspend);

        assert!(checkpoint.is_empty());
        assert!(project.typedef_params().ends_with(", uint64_t"));
        assert!(boundary.contains("if (unlikely(instret_remaining < 1u))"));
        assert!(boundary.contains("state->pc = 0x00000100ull;"));
        assert!(boundary.contains("[[clang::musttail]] return rv_suspend("));
        assert!(boundary.contains("instret_remaining -= 1u;"));
        assert!(suspend.contains("preserve_none, noinline"));
        assert!(!suspend.contains("cold"));
        assert!(suspend
            .contains("state->mode_state.retired = state->mode_state.target - instret_remaining;"));

        #[cfg(target_arch = "aarch64")]
        assert_eq!(project.hot_regs.len(), 22);
        #[cfg(not(target_arch = "aarch64"))]
        assert_eq!(project.hot_regs.len(), 9);
    }

    #[test]
    fn oversized_block_abi_is_rejected() {
        let mut project = CProject::new(
            Path::new("unused"),
            "test",
            RvrExecutionKind::PureWithInstretTracking,
        );
        project.hot_regs = CProject::hot_regs_for_kind(RvrExecutionKind::Pure);

        let error = project.validate_block_abi().unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert!(error.to_string().contains("preserve_none supports"));
    }

    #[test]
    fn metered_codegen_rejects_extension_chip_outside_air_count() {
        let mut project = CProject::new(Path::new("unused"), "test", RvrExecutionKind::Metered);
        project.pc_base = 0x100;
        project.num_airs = Some(1);
        project.pc_to_chip = Some(vec![
            rvr_openvm_lift::TraceChipIndex::NoChip,
            rvr_openvm_lift::TraceChipIndex::NoChip,
        ]);
        let block = Block {
            start_pc: 0x100,
            end_pc: 0x108,
            instructions: vec![InstrAt {
                pc: 0x100,
                instr: Box::new(InvalidTraceChipInstr),
                source_loc: None,
            }],
            terminator: Terminator::Exit { code: 0 },
            terminator_pc: 0x104,
            terminator_source_loc: None,
        };

        let error = project
            .emit_block_function(&mut String::new(), &block, &HashSet::new())
            .unwrap_err();

        assert_eq!(error.to_string(), "chip index 1 is outside AIR count 1");
    }

    #[test]
    fn metered_cost_counts_without_suspension_layer() {
        let project = CProject::new(Path::new("unused"), "test", RvrExecutionKind::MeteredCost);
        let mut boundary = String::new();
        project.emit_block_boundary(&mut boundary, &single_instruction_block());

        assert_eq!(boundary, "    state->mode_state.instret += 1u;\n");
        assert!(!project.typedef_params().contains("instret_remaining"));
    }

    #[test]
    fn execution_kinds_have_distinct_suffixes_and_round_trip_markers() {
        let kinds = [
            RvrExecutionKind::Pure,
            RvrExecutionKind::PureWithInstretTracking,
            RvrExecutionKind::MeteredCost,
            RvrExecutionKind::Metered,
            RvrExecutionKind::MeteredSegment,
        ];
        let mut suffixes = HashSet::new();
        for kind in kinds {
            assert!(suffixes.insert(kind.artifact_suffix()));
            assert_eq!(RvrExecutionKind::try_from(kind as u32), Ok(kind));
        }
        assert!(RvrExecutionKind::try_from(u32::MAX).is_err());
    }
}
