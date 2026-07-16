//! C project generation: header, block files, dispatch, Makefile.

use std::{
    collections::{BTreeMap, HashSet},
    fmt::Write,
    fs, io,
    path::{Path, PathBuf},
};

use rvr_openvm_ir::*;
use rvr_openvm_lift::{ExtensionRegistry, TraceChipIndex};

use super::{
    codegen::{emit_terminator, InstrCodegen, TermCtx},
    context::{EmitContext, EmitMode},
};
use crate::constants::constants_header;

/// Compile-time tracer selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TracerMode {
    /// Pure execution with optional instruction-count suspension.
    Pure,
    /// Scalar metered cost accumulation (MeteredCostCtx).
    MeteredCost,
    /// Per-chip trace heights (MeteredCtx).
    Metered,
    /// Preflight execution logging program and memory accesses.
    Preflight,
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
            TracerMode::Preflight => "openvm_tracer_preflight.h",
        }
    }

    fn header_content(self) -> &'static str {
        match self {
            TracerMode::Pure => include_str!("../../c/tracer/openvm_tracer_pure.h"),
            TracerMode::MeteredCost => {
                include_str!("../../c/tracer/openvm_tracer_metered_cost.h")
            }
            TracerMode::Metered => include_str!("../../c/tracer/openvm_tracer_metered.h"),
            TracerMode::Preflight => {
                include_str!("../../c/tracer/openvm_tracer_preflight_common.h")
            }
        }
    }

    pub fn default_suspend_policy(self) -> SuspendPolicy {
        match self {
            TracerMode::Pure | TracerMode::MeteredCost | TracerMode::Preflight => {
                SuspendPolicy::InstretLimit
            }
            TracerMode::Metered => SuspendPolicy::Disabled,
        }
    }

    fn block_header_content(self, suspend_policy: SuspendPolicy) -> &'static str {
        match (self, suspend_policy) {
            (TracerMode::Metered, SuspendPolicy::SegmentBoundary) => {
                include_str!("../../c/block/openvm_block_metered_segment.h")
            }
            (TracerMode::Metered, _) => include_str!("../../c/block/openvm_block_metered.h"),
            (TracerMode::Pure | TracerMode::MeteredCost | TracerMode::Preflight, _) => {
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct G2DsoManifestConfigV1 {
    pub fingerprint: [u8; 32],
    pub program_fingerprint: [u8; 32],
    pub block_fingerprint: [u8; 32],
    pub air_manifest_fingerprint: [u8; 32],
    pub pc_base: u32,
    pub block_count: u32,
    pub air_count: u32,
    pub air_kinds: [u8; 31],
    pub air_indices: [u32; 31],
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
    /// Per-program-slot filtered execution-frequency index (`u32::MAX` for
    /// holes). ZG2 bakes this into each preflight `trace_pc_indexed` call.
    pub pc_to_exec_idx: Vec<u32>,
    /// Program PC base (used to compute pc_to_chip index).
    pub pc_base: u64,
    /// Per-AIR widths for MeteredCost precomputation. Indexed by chip index.
    pub chip_widths: Option<Vec<u64>>,
    /// Compile with native debug info (`-g -fno-omit-frame-pointer`).
    pub native_debug_info: bool,
    /// Emit profiling-only hot-loop hooks. Normal projects compile them away.
    pub native_detail: bool,
    /// Profiling family for each program slot, derived from the original VM
    /// opcode before extension lifting erases that distinction.
    pub native_detail_pc_families: Vec<u8>,
    /// R3: emit inline compact records (log-suppressed) for migrated opcodes.
    /// Preflight mode only; see `inline_records_enabled`.
    pub inline_records: bool,
    /// Effective per-program-slot inline decision after compiler-level
    /// whole-AIR taint. This, rather than the IR node's shape alone, controls
    /// program-log suppression and delta reservation.
    pub inline_pc_slots: Vec<bool>,
    /// Stage-2 chronological 24-byte delta stream. This is a stronger compact
    /// mode: all inline AIRs share one execution-ordered backing and reserve
    /// their record slots once per basic-block entry.
    pub delta_records: bool,
    /// Private G2 v1 lane producer for the currently negotiated compact
    /// families. Every existing route remains unchanged when this is false.
    pub g2_records: bool,
    /// Stable decoder kind per program slot. G2 codegen uses this to select
    /// the final V0/V1 lane without re-deriving opcode families in C.
    pub g2_pc_kinds: Vec<u8>,
    /// Full schema/program/AIR binding exported by the generated DSO.
    pub g2_manifest: Option<G2DsoManifestConfigV1>,
    /// R4: airs whose records the generated C writes arena-native — full
    /// records at final arena positions, field offsets baked as literals
    /// from the geometry's layout table. Airs absent here keep the compact
    /// wire. Populated by the host compile pipeline from the assembler
    /// registry; empty means pure R3 emission.
    pub arena_native_airs: std::collections::BTreeMap<u32, crate::ArenaNativeGeometry>,
}

impl CProject {
    pub fn new(output_dir: &Path, name: &str, tracer_mode: TracerMode) -> Self {
        // Hot registers in priority order.
        // Limited by platform's preserve_none register capacity minus 1 (state ptr).
        let hot_regs = Self::hot_regs_for_mode(tracer_mode);

        Self {
            output_dir: output_dir.to_path_buf(),
            name: name.to_string(),
            tracer_mode,
            suspend_policy: tracer_mode.default_suspend_policy(),
            hot_regs,
            blocks_per_partition: 512,
            enable_lto: true,
            pc_to_chip: None,
            pc_to_exec_idx: Vec::new(),
            pc_base: 0,
            chip_widths: None,
            native_debug_info: false,
            native_detail: false,
            native_detail_pc_families: Vec::new(),
            inline_records: false,
            inline_pc_slots: Vec::new(),
            delta_records: false,
            g2_records: false,
            g2_pc_kinds: Vec::new(),
            g2_manifest: None,
            arena_native_airs: std::collections::BTreeMap::new(),
        }
    }

    /// Register priority order.
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

    /// Metered mode carries `check_counter` and `trace_heights` through the
    /// preserve_none ABI, so it reserves two argument registers that would
    /// otherwise hold guest regs.
    fn hot_regs_for_mode(mode: TracerMode) -> HashSet<u8> {
        match mode {
            TracerMode::Metered => {
                #[cfg(target_arch = "aarch64")]
                const NUM_HOT_REGS: usize = 21;
                #[cfg(not(target_arch = "aarch64"))]
                const NUM_HOT_REGS: usize = 8;

                Self::REG_PRIORITY[..NUM_HOT_REGS].iter().copied().collect()
            }
            TracerMode::Pure | TracerMode::MeteredCost | TracerMode::Preflight => {
                if mode == TracerMode::Preflight {
                    HashSet::new()
                } else {
                    Self::default_hot_regs()
                }
            }
        }
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

    fn append_metered_args_from_state(&self, out: &mut String) {
        if self.tracer_mode == TracerMode::Metered {
            out.push_str(", state->tracer->check_counter, state->tracer->trace_heights");
        }
    }

    fn append_metered_args_from_params(&self, out: &mut String) {
        if self.tracer_mode == TracerMode::Metered {
            out.push_str(", check_counter, trace_heights");
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
    fn param_list_items(&self, include_names: bool) -> Vec<String> {
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
        if self.tracer_mode == TracerMode::Metered {
            params.push(if include_names {
                "uint32_t check_counter".to_string()
            } else {
                "uint32_t".to_string()
            });
            params.push(if include_names {
                "uint32_t* trace_heights".to_string()
            } else {
                "uint32_t*".to_string()
            });
        }
        params
    }

    fn block_signature(&self, prefix: &str, name: &str) -> String {
        let params = self.param_list_items(true);
        let mut out = format!("{prefix} {name}(\n");
        for (idx, param) in params.iter().enumerate() {
            let suffix = if idx + 1 == params.len() { "" } else { "," };
            writeln!(out, "    {param}{suffix}").unwrap();
        }
        out.push(')');
        out
    }

    /// C argument list extracting hot regs from state:
    /// "state, state->regs[1], state->regs[2]".
    fn fn_args_from_state(&self) -> String {
        let mut s = "state".to_string();
        self.append_hot_reg_args_from_state(&mut s);
        self.append_metered_args_from_state(&mut s);
        s
    }

    /// C argument list forwarding the current block ABI parameters.
    fn fn_args_from_params(&self) -> String {
        let mut s = "state".to_string();
        self.append_hot_reg_args_from_params(&mut s);
        self.append_metered_args_from_params(&mut s);
        s
    }

    /// C typedef parameter types: "RvState*, uint32_t, uint32_t".
    fn typedef_params(&self) -> String {
        self.param_list_items(false).join(", ")
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
        match self.tracer_mode {
            TracerMode::Pure => EmitMode::Direct,
            TracerMode::Metered => EmitMode::Metered {
                trace_memory_pages: block_accesses_memory(block),
            },
            TracerMode::MeteredCost => EmitMode::Direct,
            TracerMode::Preflight => EmitMode::ValueTrace,
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

    fn exec_idx_for_pc(&self, pc: u64) -> u32 {
        let slot = ((pc - self.pc_base) / 4) as usize;
        *self
            .pc_to_exec_idx
            .get(slot)
            .unwrap_or_else(|| panic!("pc {pc:#x} outside execution-frequency map"))
    }

    fn pc_emits_inline_record(&self, pc: u64) -> bool {
        let Some(offset) = pc.checked_sub(self.pc_base) else {
            return false;
        };
        self.inline_pc_slots
            .get((offset / 4) as usize)
            .copied()
            .unwrap_or(false)
    }

    fn g2_kind_for_pc(&self, pc: u64) -> u8 {
        let Some(offset) = pc.checked_sub(self.pc_base) else {
            return u8::MAX;
        };
        self.g2_pc_kinds
            .get((offset / 4) as usize)
            .copied()
            .unwrap_or(u8::MAX)
    }

    fn native_detail_family(&self, pc: u64) -> u32 {
        let Some(offset) = pc.checked_sub(self.pc_base) else {
            return 8;
        };
        self.native_detail_pc_families
            .get((offset / 4) as usize)
            .copied()
            .unwrap_or(8) as u32
    }

    /// Write all C project files.
    pub fn write_all(
        &self,
        blocks: &[Block],
        entry_point: u64,
        text_start: u64,
        extensions: &ExtensionRegistry,
    ) -> io::Result<()> {
        let text_end = Self::dispatch_max_pc(blocks, entry_point, text_start);
        let table_size = Self::dispatch_table_size(text_start, text_end);

        self.write_constants(text_start, text_end, table_size)?;
        self.write_g2_manifest()?;
        self.write_support_files()?;
        self.write_extension_files(extensions)?;
        let ext_headers = extensions.c_headers();
        self.write_header(blocks, &ext_headers)?;
        self.write_block_files(blocks)?;
        self.write_dispatch(blocks, entry_point, text_start)?;
        self.write_makefile()?;
        Ok(())
    }

    fn write_g2_manifest(&self) -> io::Result<()> {
        let path = self.output_dir.join("openvm_g2_manifest.c");
        let Some(manifest) = self.g2_manifest else {
            let _ = fs::remove_file(path);
            return Ok(());
        };
        let fingerprint = manifest
            .fingerprint
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let program_fingerprint = manifest
            .program_fingerprint
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let block_fingerprint = manifest
            .block_fingerprint
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let air_manifest_fingerprint = manifest
            .air_manifest_fingerprint
            .iter()
            .map(|byte| format!("0x{byte:02x}"))
            .collect::<Vec<_>>()
            .join(", ");
        let air_kinds = manifest
            .air_kinds
            .iter()
            .map(|kind| format!("{kind}u"))
            .collect::<Vec<_>>()
            .join(", ");
        let air_indices = manifest
            .air_indices
            .iter()
            .map(|index| format!("{index}u"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut lanes = vec![(0x0001u16, 4u8, 1u32, 0u32, 0u8)];
        lanes.extend([
            (0x0080, 8, 3, 2, 1),
            (0x0081, 1, 3, 2, 1),
            (0x0082, 8, 3, 2, 1),
            (0x0083, 4, 3, 2, 1),
        ]);
        for kind in 0u8..30 {
            for value_lane in [false, true] {
                let Some(width) =
                    rvr_openvm_ext_ffi_common::g2_standard_lane_width(kind, value_lane)
                else {
                    continue;
                };
                let load_store = rvr_openvm_ext_ffi_common::G2_LOAD_STORE_KINDS.contains(&kind);
                lanes.push((
                    if value_lane {
                        rvr_openvm_ext_ffi_common::g2_lane_v1(kind)
                    } else {
                        rvr_openvm_ext_ffi_common::g2_lane_v0(kind)
                    },
                    width,
                    if load_store { 3 } else { 1 },
                    if load_store { 1 } else { 0 },
                    match kind {
                        13 | 29 => 1,
                        1..=7 => u8::MAX,
                        _ => 2,
                    },
                ));
            }
        }
        lanes.sort_unstable_by_key(|lane| lane.0);
        let lane_source = lanes
            .iter()
            .map(|&(kind, width, flags, group, arity)| {
                format!(
                    "                 {{.kind=0x{kind:04x}, .elem_width={width}, .encoding=0, .flags={flags}, .group_id={group}, .arity={arity}, .reserved={{0,0,0}}}},"
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let source = format!(
            "#include <stdint.h>\n\n\
             typedef struct OpenVmRvrG2DsoLaneManifestV1 {{\n\
               uint16_t kind;\n\
               uint8_t elem_width;\n\
               uint8_t encoding;\n\
               uint32_t flags;\n\
               uint32_t group_id;\n\
               uint8_t arity;\n\
               uint8_t reserved[3];\n\
             }} OpenVmRvrG2DsoLaneManifestV1;\n\n\
             typedef struct OpenVmRvrG2DsoManifestV1 {{\n\
               uint8_t magic[8];\n\
               uint16_t version;\n\
               uint16_t manifest_bytes;\n\
               uint16_t header_size;\n\
               uint16_t lane_desc_size;\n\
               uint32_t lane_count;\n\
               uint32_t wire_flags;\n\
               uint8_t fingerprint[32];\n\
               uint8_t program_fingerprint[32];\n\
               uint8_t block_fingerprint[32];\n\
               uint8_t air_manifest_fingerprint[32];\n\
               uint32_t pc_base;\n\
               uint32_t block_count;\n\
               uint32_t air_count;\n\
               uint32_t reserved;\n\
               uint8_t air_kinds[31];\n\
               uint32_t air_indices[31];\n\
               OpenVmRvrG2DsoLaneManifestV1 lanes[59];\n\
             }} OpenVmRvrG2DsoManifestV1;\n\n\
             _Static_assert(sizeof(OpenVmRvrG2DsoLaneManifestV1) == 16, \"G2 DSO lane manifest size drift\");\n\
             _Static_assert(sizeof(OpenVmRvrG2DsoManifestV1) == 1268, \"G2 DSO manifest size drift\");\n\n\
             __attribute__((visibility(\"default\")))\n\
             const OpenVmRvrG2DsoManifestV1 openvm_rvr_g2_manifest_v1 = {{\n\
               .magic = {{'O','V','M','G','2','D','1','\\0'}},\n\
               .version = 1,\n\
               .manifest_bytes = 1268,\n\
               .header_size = 64,\n\
               .lane_desc_size = 32,\n\
               .lane_count = 59,\n\
               .wire_flags = 14,\n\
               .fingerprint = {{{fingerprint}}},\n\
               .program_fingerprint = {{{program_fingerprint}}},\n\
               .block_fingerprint = {{{block_fingerprint}}},\n\
               .air_manifest_fingerprint = {{{air_manifest_fingerprint}}},\n\
               .pc_base = {},\n\
               .block_count = {},\n\
               .air_count = {},\n\
               .reserved = 0,\n\
               .air_kinds = {{{air_kinds}}},\n\
               .air_indices = {{{air_indices}}},\n\
               .lanes = {{\n\
{lane_source}\n\
               }},\n\
             }};\n",
            manifest.pc_base, manifest.block_count, manifest.air_count,
        );
        fs::write(path, source)
    }

    // ── Generated constants header ──────────────────────────────────────

    fn write_constants(
        &self,
        text_start: u64,
        text_end: u64,
        dispatch_table_size: usize,
    ) -> io::Result<()> {
        let mut h = constants_header(text_start, text_end, dispatch_table_size);
        writeln!(
            h,
            "static constexpr bool OPENVM_RVR_NATIVE_DETAIL_ENABLED = {};",
            if self.native_detail { "true" } else { "false" }
        )
        .unwrap();
        let path = self.output_dir.join("openvm_constants.h");
        fs::write(&path, h)
    }

    // ── Support files (tracer header, state header, IO) ─────────────────

    fn write_support_files(&self) -> io::Result<()> {
        fs::write(
            self.output_dir.join("openvm_util.h"),
            include_str!("../../c/openvm_util.h"),
        )?;

        // RvState struct definition (forward-declares Tracer).
        let state_path = self.output_dir.join("openvm_state.h");
        fs::write(&state_path, include_str!("../../c/openvm_state.h"))?;

        // Tracer header (includes openvm_state.h, defines Tracer + inline functions).
        let tracer_path = self.output_dir.join(self.tracer_mode.header_filename());
        if self.tracer_mode == TracerMode::Preflight {
            fs::write(
                self.output_dir.join("openvm_tracer_preflight_common.h"),
                include_str!("../../c/tracer/openvm_tracer_preflight_common.h"),
            )?;
            if self.delta_records {
                fs::write(
                    self.output_dir.join("openvm_tracer_preflight_delta.h"),
                    include_str!("../../c/tracer/openvm_tracer_preflight_delta.h"),
                )?;
            }
            if self.native_detail {
                fs::write(
                    self.output_dir
                        .join("openvm_tracer_preflight_native_detail.h"),
                    include_str!("../../c/tracer/openvm_tracer_preflight_native_detail.h"),
                )?;
            }

            let mut route_header = String::from(
                "#ifndef OPENVM_TRACER_PREFLIGHT_H\n#define OPENVM_TRACER_PREFLIGHT_H\n\n",
            );
            if self.delta_records {
                route_header.push_str("#define OPENVM_RVR_PREFLIGHT_DELTA 1\n");
            }
            if self.native_detail {
                route_header.push_str("#define OPENVM_RVR_PREFLIGHT_NATIVE_DETAIL 1\n");
            }
            route_header.push_str(
                "#include \"openvm_tracer_preflight_common.h\"\n\n#endif /* OPENVM_TRACER_PREFLIGHT_H */\n",
            );
            fs::write(&tracer_path, route_header)?;
        } else {
            fs::write(&tracer_path, self.tracer_mode.header_content())?;
        }

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

    fn write_extension_files(&self, extensions: &ExtensionRegistry) -> io::Result<()> {
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

    fn write_header(&self, blocks: &[Block], ext_headers: &[(&str, &str)]) -> io::Result<()> {
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
        writeln!(h, "extern BlockFn dispatch_table[RV_DISPATCH_TABLE_SIZE];").unwrap();
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
        let valid_blocks: HashSet<u64> = blocks.iter().map(|b| b.start_pc).collect();

        for part_idx in 0..num_partitions {
            let start = part_idx * self.blocks_per_partition;
            let end = (start + self.blocks_per_partition).min(blocks.len());
            let partition = &blocks[start..end];

            let first_pc = partition[0].start_pc;
            let mut src = String::with_capacity(64 * 1024);
            writeln!(src, "#include \"{name}.h\"").unwrap();
            writeln!(src).unwrap();

            for block in partition {
                self.emit_block_checkpoint_function(&mut src, block);
                self.emit_block_function(&mut src, block, &valid_blocks);
            }

            let path = self.output_dir.join(format!("{name}_0x{first_pc:08x}.c"));
            fs::write(&path, src)?;
        }

        Ok(())
    }

    fn emit_block_function(&self, out: &mut String, block: &Block, valid_blocks: &HashSet<u64>) {
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
        let signature = self.block_signature(
            "__attribute__((preserve_none)) void",
            &format!("block_0x{pc:08x}"),
        );
        writeln!(out, "{signature} {{").unwrap();

        // Body instructions (each in its own scope to avoid variable collisions).
        let mut ctx = EmitContext::new(self.hot_regs.clone(), mode);
        let inline_records = self.inline_records_enabled();
        ctx.set_inline_records(inline_records);
        if inline_records && !self.arena_native_airs.is_empty() {
            ctx.set_arena_native_airs(self.arena_native_airs.clone());
        }

        writeln!(out, "    uint8_t* memory = state->memory;").unwrap();

        let delta_count = if self.delta_records && inline_records {
            let body = block
                .instructions
                .iter()
                .filter(|instr_at| {
                    self.pc_emits_inline_record(instr_at.pc)
                        && matches!(
                            self.chip_idx_for_pc(instr_at.pc),
                            TraceChipIndex::Chip(air)
                                if !self.arena_native_airs.contains_key(&air.as_u32())
                        )
                })
                .count();
            let terminator = (!matches!(block.terminator, Terminator::FallThrough)
                && self.pc_emits_inline_record(block.terminator_pc)
                && matches!(
                    self.chip_idx_for_pc(block.terminator_pc),
                    TraceChipIndex::Chip(air)
                        if !self.arena_native_airs.contains_key(&air.as_u32())
                )) as usize;
            body + terminator
        } else {
            0
        };
        let delta_batch = (delta_count != 0).then(|| "rvr_delta_batch".to_string());

        if matches!(
            mode,
            EmitMode::Metered {
                trace_memory_pages: true
            }
        ) {
            writeln!(
                out,
                "    TraceMemory trace_memory = trace_memory_setup(state->tracer);"
            )
            .unwrap();
        }

        self.emit_block_boundary(out, block);
        if let Some(batch) = delta_batch.as_deref() {
            writeln!(
                out,
                "    PreflightDeltaRecord* {batch} = preflight_claim_delta_records(state, {delta_count}u);"
            )
            .unwrap();
        }
        ctx.set_delta_records(self.delta_records, delta_batch);
        ctx.set_g2_records(self.g2_records);
        self.emit_per_block_chip_updates(out, block);

        for instr_at in &block.instructions {
            self.emit_source_annotation(
                out,
                instr_at.pc,
                instr_at.instr.opname(),
                instr_at.source_loc.as_ref(),
            );
            if inline_records {
                let chip_idx = match (
                    self.pc_emits_inline_record(instr_at.pc),
                    self.chip_idx_for_pc(instr_at.pc),
                ) {
                    (true, TraceChipIndex::Chip(air)) => air.as_u32(),
                    _ => u32::MAX,
                };
                ctx.set_current_instr(chip_idx, instr_at.pc, self.g2_kind_for_pc(instr_at.pc));
            }
            ctx.trace_pc(
                instr_at.pc,
                self.exec_idx_for_pc(instr_at.pc),
                inline_records && self.pc_emits_inline_record(instr_at.pc),
                self.native_detail_family(instr_at.pc),
            );
            instr_at.instr.emit_c(&mut ctx);
            Self::emit_context_scope(out, &mut ctx);
            out.push('\n');
        }

        ctx.flush_page_locals();
        Self::emit_context_scope(out, &mut ctx);

        let has_terminator_instruction = !matches!(block.terminator, Terminator::FallThrough);
        if has_terminator_instruction {
            self.emit_source_annotation(
                out,
                block.terminator_pc,
                block.terminator.opname(),
                block.terminator_source_loc.as_ref(),
            );
            if inline_records {
                let chip_idx = match (
                    self.pc_emits_inline_record(block.terminator_pc),
                    self.chip_idx_for_pc(block.terminator_pc),
                ) {
                    (true, TraceChipIndex::Chip(air)) => air.as_u32(),
                    _ => u32::MAX,
                };
                ctx.set_current_instr(
                    chip_idx,
                    block.terminator_pc,
                    self.g2_kind_for_pc(block.terminator_pc),
                );
            }
            ctx.trace_pc(
                block.terminator_pc,
                self.exec_idx_for_pc(block.terminator_pc),
                inline_records && self.pc_emits_inline_record(block.terminator_pc),
                self.native_detail_family(block.terminator_pc),
            );
        }
        let tc = TermCtx { valid_blocks };
        emit_terminator(&mut ctx, &block.terminator, block.terminator_pc, &tc);
        Self::emit_context_scope(out, &mut ctx);

        writeln!(out, "}}").unwrap();
        writeln!(out).unwrap();
    }

    fn emit_block_checkpoint_function(&self, out: &mut String, block: &Block) {
        if self.tracer_mode != TracerMode::Metered {
            return;
        }

        let pc = block.start_pc;
        let args = self.fn_args_from_params();

        let signature = self.block_signature(
            "static __attribute__((preserve_none, cold, noinline)) void",
            &format!("block_0x{pc:08x}_checkpoint"),
        );
        writeln!(out, "{signature} {{").unwrap();
        match self.suspend_policy {
            SuspendPolicy::SegmentBoundary => {
                self.emit_segment_checkpoint(out, pc);
            }
            _ => {
                writeln!(
                    out,
                    "    check_counter = metered_checkpoint(state, check_counter);"
                )
                .unwrap();
            }
        }
        writeln!(
            out,
            "    [[clang::musttail]] return block_0x{pc:08x}({args});"
        )
        .unwrap();
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
        writeln!(out, "    if (unlikely(checkpoint.suspend_signal)) {{").unwrap();
        self.emit_suspend_return(out, pc);
        writeln!(out, "    }}").unwrap();
    }

    fn emit_block_boundary(&self, out: &mut String, block: &Block) {
        let pc = block.start_pc;
        let insn_count = block.insn_count();
        let is_metered = self.tracer_mode == TracerMode::Metered;

        if is_metered {
            self.emit_metered_counter_check(out, pc, insn_count);
            writeln!(out, "    check_counter -= {insn_count}u;").unwrap();
            return;
        }

        writeln!(
            out,
            "    uint8_t suspend_signal = begin_block(state, 0x{pc:08x}ull, {insn_count}u);"
        )
        .unwrap();
        self.emit_instret_suspend_check(out, pc, insn_count);
        if self.delta_records || self.g2_records {
            // Commit the block-level trace only after an instret-limit
            // rejection has returned. Device chronology treats this hook as
            // an exact executed-block stream; recording it in begin_block
            // would also include the first unexecuted block of every
            // continuation boundary.
            if self.delta_records {
                writeln!(out, "    trace_block(state, 0x{pc:08x}ull, {insn_count}u);").unwrap();
            }
            if self.g2_records {
                let slot = (pc - self.pc_base) / 4;
                writeln!(
                    out,
                    "    preflight_g2_emit_run(state, {slot}u, {insn_count}u);"
                )
                .unwrap();
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

    fn emit_instret_suspend_check(&self, out: &mut String, pc: u64, insn_count: u32) {
        debug_assert_ne!(
            self.tracer_mode,
            TracerMode::Metered,
            "metered instret-limit suspension is rejected in compile options"
        );
        writeln!(
            out,
            "    if (unlikely(should_suspend(state, 0x{pc:08x}ull, {insn_count}u, suspend_signal))) {{"
        )
        .unwrap();
        self.emit_suspend_return(out, pc);
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
    /// batched tracer update instead of per-instruction `trace_chip` calls.
    ///
    /// Mode-dependent emission:
    ///   - Pure: nothing (chip tracking is a no-op).
    ///   - Metered: `trace_heights[idx] += count;` per distinct chip, using the trace-heights
    ///     pointer carried through the block ABI.
    ///   - MeteredCost: `state->tracer->cost += <constant>;` where the constant is `sum(width[chip]
    ///     * count)` precomputed at emit time from `self.chip_widths`.
    ///   - Preflight: `trace_chip(state, idx, count);` per distinct chip.
    fn emit_per_block_chip_updates(&self, out: &mut String, block: &Block) {
        if matches!(self.tracer_mode, TracerMode::Pure) {
            return;
        }

        let mut chip_counts: BTreeMap<u32, u32> = BTreeMap::new();
        let mut increment_chip_count = |pc: u64| match self.chip_idx_for_pc(pc) {
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
                for (chip, count) in &chip_counts {
                    writeln!(out, "        trace_heights[{chip}] += {count}u;").unwrap();
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
            TracerMode::Preflight => {
                for (chip, count) in &chip_counts {
                    writeln!(out, "        trace_chip(state, {chip}u, {count}u);").unwrap();
                }
            }
        }
        writeln!(out, "    }}").unwrap();
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

        let save = self.save_hot_regs_call();
        // rv_trap — cold fallback for dispatch to non-block PCs.
        let trap_signature =
            self.block_signature("__attribute__((preserve_none, cold)) void", "rv_trap");
        writeln!(src, "{trap_signature} {{").unwrap();
        writeln!(src, "    {save}").unwrap();
        if self.tracer_mode == TracerMode::Metered {
            writeln!(src, "    state->tracer->check_counter = check_counter;").unwrap();
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

        writeln!(src, "BlockFn dispatch_table[RV_DISPATCH_TABLE_SIZE] = {{").unwrap();
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

        // rv_execute — single entry call; tail calls chain blocks.
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

fn instr_accesses_memory(instr: &Instr) -> bool {
    match instr {
        Instr::Load { .. } | Instr::Store { .. } => true,
        Instr::Ext(ext) => ext.accesses_memory(),
        _ => false,
    }
}

fn terminator_accesses_memory(terminator: &Terminator) -> bool {
    match terminator {
        Terminator::Extension(ext) => ext.accesses_memory(),
        _ => false,
    }
}

fn block_accesses_memory(block: &Block) -> bool {
    block
        .instructions
        .iter()
        .any(|instr_at| instr_accesses_memory(&instr_at.instr))
        || terminator_accesses_memory(&block.terminator)
}
