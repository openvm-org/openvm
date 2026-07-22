//! Extension registry for plugging in new opcode families.

use std::{
    any::type_name,
    collections::{HashMap, HashSet},
};

use openvm_instructions::{exe::SparseMemoryImage, LocalOpcode, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::{FixedTraceRows, LiftedInstr, ValueSlot};

use crate::RvrInstruction;

/// Real AIR index in the trace.
///
/// `Option<AirIndex>` represents an extension's dynamic tracing metadata: `None`
/// in pure mode (AIR metadata was not requested), `Some(idx)` in metered modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AirIndex(u32);

impl AirIndex {
    #[inline]
    pub const fn new(idx: u32) -> Self {
        Self(idx)
    }

    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    #[inline]
    pub fn next(self) -> AirIndex {
        AirIndex(self.0 + 1)
    }
}

/// PC-to-chip mapping entry. `NoChip` means the instruction at this PC
/// intentionally does not contribute to any chip's trace (e.g. `TERMINATE` or
/// unmapped slots).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TraceChipIndex {
    Chip(AirIndex),
    NoChip,
}

/// Lower an extension's `Option<AirIndex>` to the `u32` chip index baked into
/// generated C source. `None` is expected only in pure mode, where tracing is a
/// no-op. Metered modes must resolve real chip indices before calling
/// chip-tracing helpers.
#[inline]
pub fn air_index_to_c(idx: Option<AirIndex>) -> u32 {
    idx.map_or(u32::MAX, AirIndex::as_u32)
}

/// Return extra fixed rows when this extension has an AIR index.
#[inline]
pub fn fixed_trace_rows_for_chip(idx: Option<AirIndex>, count: u32) -> Vec<FixedTraceRows> {
    idx.map(|idx| {
        vec![FixedTraceRows {
            chip_idx: idx.as_u32(),
            count,
        }]
    })
    .unwrap_or_default()
}

/// Decodes an aligned OpenVM operand into an opaque value slot.
pub fn decode_value_slot(value: u32, stride: u32, slot_count: u32) -> ValueSlot {
    assert!(stride != 0, "value-slot stride must be nonzero");
    assert_eq!(value % stride, 0, "value-slot operand must be slot-aligned");
    let index = value / stride;
    assert!(index < slot_count, "value-slot operand is out of bounds");
    ValueSlot::new(index)
}

/// Data needed by extension crates to resolve opcode/chip metadata when
/// registering rvr extension handlers.
#[derive(Clone, Debug, Default)]
pub struct RvrExtensionCtx {
    /// `opcode -> executor_idx` mapping.
    pub opcode_to_executor_idx: HashMap<VmOpcode, usize>,
    /// `executor_idx -> air_idx` mapping.
    pub executor_idx_to_air_idx: Vec<usize>,
}

impl RvrExtensionCtx {
    pub fn new(
        opcode_to_executor_idx: impl IntoIterator<Item = (VmOpcode, usize)>,
        executor_idx_to_air_idx: Vec<usize>,
    ) -> Self {
        Self {
            opcode_to_executor_idx: opcode_to_executor_idx.into_iter().collect(),
            executor_idx_to_air_idx,
        }
    }

    pub fn resolve_opcode_executor_idx(&self, opcode: VmOpcode) -> Option<usize> {
        self.opcode_to_executor_idx.get(&opcode).copied()
    }
}

/// Resolve the AIR index for `opcode`. In pure mode (`ctx = None`) returns
/// `Ok(None)`; in metered mode returns `Ok(Some(idx))` for the registered
/// opcode and errors if the opcode is unknown.
pub fn opcode_air_idx(
    ctx: Option<&RvrExtensionCtx>,
    opcode: impl LocalOpcode,
) -> Result<Option<AirIndex>, ExtensionError> {
    let opcode = opcode.global_opcode();
    let Some(ctx) = ctx else {
        return Ok(None);
    };
    let executor_idx = ctx
        .resolve_opcode_executor_idx(opcode)
        .ok_or(ExtensionError::UnknownOpcode(opcode))?;
    let raw = ctx.executor_idx_to_air_idx.get(executor_idx).ok_or(
        ExtensionError::ExecutorIndexOutOfBounds {
            opcode,
            executor_idx,
        },
    )?;
    let air_idx = u32::try_from(*raw).map_err(|_| ExtensionError::AirIndexOutOfBounds {
        opcode,
        air_idx: *raw,
    })?;
    Ok(Some(AirIndex::new(air_idx)))
}

/// Errors raised when resolving extension metadata from a `RvrExtensionCtx`.
#[derive(Debug, thiserror::Error)]
pub enum ExtensionError {
    #[error("opcode {0:?} not found in rvr extension context mappings")]
    UnknownOpcode(VmOpcode),
    #[error(
        "executor index {executor_idx} for opcode {opcode:?} is out of bounds in \
         executor_idx_to_air_idx"
    )]
    ExecutorIndexOutOfBounds {
        opcode: VmOpcode,
        executor_idx: usize,
    },
    #[error("AIR index {air_idx} for opcode {opcode:?} does not fit in u32")]
    AirIndexOutOfBounds { opcode: VmOpcode, air_idx: usize },
    #[error("failed to register host callbacks: {0}")]
    HostCallbackRegistration(String),
    #[error(
        "opcode {opcode:?} at pc {pc:#x} was claimed by both {first_extension} and {second_extension}"
    )]
    DuplicateOpcodeClaim {
        opcode: VmOpcode,
        pc: u64,
        first_extension: &'static str,
        second_extension: &'static str,
    },
}

/// Trait for an rvr-openvm extension. Each extension handles a range of opcodes
/// and knows how to lift them to IR (with self-describing codegen via `ExtInstr::emit_c`).
pub trait RvrExtension: Send + Sync {
    /// Try to lift an OpenVM instruction into IR.
    /// Return `None` if this extension doesn't handle the opcode.
    /// Chip indices are stored on the extension and baked into IR nodes.
    fn try_lift(&self, insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr>;

    /// C header files for this extension, as `(filename, content)` pairs.
    /// Written to the output directory and `#include`d in the generated code.
    fn c_headers(&self) -> Vec<(&'static str, &'static str)>;

    /// C source files for this extension, as `(filename, content)` pairs.
    /// Written to the output directory and compiled alongside the generated
    /// code. This lets extension code call static inline tracer helpers
    /// directly instead of routing through separate Rust FFI wrappers.
    fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Embedded pre-built static libraries (.a) for this extension.
    /// These are written to the generated project and linked into the final
    /// .so/.dylib. Default is empty (extension has no native side-car).
    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        Vec::new()
    }

    /// Whether this extension's static library calls the shared memory wrappers.
    fn uses_memory_wrappers(&self) -> bool {
        false
    }

    /// Maximum number of main-memory pages one instruction can add to the
    /// metering buffer.
    fn max_main_memory_pages_per_instruction(&self) -> usize {
        0
    }

    /// Returns instruction targets encoded in target-specific initialized data.
    fn extra_cfg_targets(
        &self,
        _init_memory: &SparseMemoryImage,
        _valid_pcs: &HashSet<u64>,
    ) -> Vec<u64> {
        Vec::new()
    }

    /// Third-party C source files compiled separately.
    ///
    /// OpenVM warning flags do not apply to these files. If OpenVM-owned C
    /// includes a third-party file, provide it through
    /// [`Self::extra_c_include_files`] and use an `-isystem` include path.
    fn vendored_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Additional embedded C files to write alongside the generated project
    /// because they are included by extension sources, but not compiled
    /// directly as translation units.
    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Extra compiler flags for extension C, such as include paths for
    /// submodule headers. The Makefile receives them as `EXT_CFLAGS`.
    fn extra_cflags(&self) -> Vec<String> {
        vec![]
    }
}

/// Installs host callbacks for an rvr extension.
pub trait RvrRuntimeExtension: Send + Sync {
    /// Register host-side callbacks with the loaded `.so`. Called after the IO
    /// context is installed and before `rv_execute`.
    ///
    /// # Safety
    ///
    /// `lib` must be the rvr-compiled shared library containing the matching
    /// extension code.
    unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError>;
}

/// Trait implemented by OpenVM extension owner types to contribute their rvr
/// lifting/codegen extensions during config assembly.
pub trait VmRvrExtension<F: PrimeField32> {
    fn extend_rvr(&self, _extensions: &mut RvrExtensions, _ctx: Option<&RvrExtensionCtx>) {}
}

impl<F: PrimeField32, EXT: VmRvrExtension<F>> VmRvrExtension<F> for Option<EXT> {
    fn extend_rvr(&self, extensions: &mut RvrExtensions, ctx: Option<&RvrExtensionCtx>) {
        if let Some(ext) = self {
            ext.extend_rvr(extensions, ctx);
        }
    }
}

// ── Extension registry ───────────────────────────────────────────────────────

/// Registry of extensions, consulted during lifting and project generation.
#[derive(Default)]
pub struct ExtensionRegistry {
    extensions: Vec<RegisteredExtension>,
}

struct RegisteredExtension {
    name: &'static str,
    extension: Box<dyn RvrExtension>,
}

impl ExtensionRegistry {
    /// Create an empty registry (no extensions).
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an extension.
    pub fn register<E: RvrExtension + 'static>(&mut self, ext: E) {
        self.extensions.push(RegisteredExtension {
            name: type_name::<E>(),
            extension: Box::new(ext),
        });
    }

    /// Try to lift an instruction through all registered extensions.
    /// Returns an error if multiple extensions claim the same instruction.
    pub fn try_lift(
        &self,
        insn: &RvrInstruction,
        pc: u64,
    ) -> Result<Option<LiftedInstr>, ExtensionError> {
        let mut lifted: Option<(&'static str, LiftedInstr)> = None;
        for registered in &self.extensions {
            let Some(candidate) = registered.extension.try_lift(insn, pc) else {
                continue;
            };
            if let Some((first_extension, _)) = &lifted {
                return Err(ExtensionError::DuplicateOpcodeClaim {
                    opcode: insn.opcode,
                    pc,
                    first_extension,
                    second_extension: registered.name,
                });
            }
            lifted = Some((registered.name, candidate));
        }
        Ok(lifted.map(|(_, instruction)| instruction))
    }

    /// Collect all C headers from all registered extensions.
    pub fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.c_headers())
            .collect()
    }

    /// Collect all C source files from all registered extensions.
    pub fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.c_sources())
            .collect()
    }

    /// Collect all embedded static libraries for linking.
    pub fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.staticlib_files())
            .collect()
    }

    /// Whether any registered extension needs the shared memory wrappers.
    pub fn uses_memory_wrappers(&self) -> bool {
        self.extensions
            .iter()
            .any(|ext| ext.extension.uses_memory_wrappers())
    }

    /// Maximum per-instruction main-memory page contribution across extensions.
    pub fn max_main_memory_pages_per_instruction(&self) -> usize {
        self.extensions
            .iter()
            .map(|ext| ext.extension.max_main_memory_pages_per_instruction())
            .max()
            .unwrap_or(0)
    }

    /// Collect target-specific CFG roots from initialized memory.
    pub fn extra_cfg_targets(
        &self,
        init_memory: &SparseMemoryImage,
        valid_pcs: &HashSet<u64>,
    ) -> Vec<u64> {
        let mut targets = self
            .extensions
            .iter()
            .flat_map(|ext| ext.extension.extra_cfg_targets(init_memory, valid_pcs))
            .collect::<Vec<_>>();
        targets.sort_unstable();
        targets.dedup();
        targets
    }

    /// Collect third-party C source files from all extensions.
    pub fn vendored_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.vendored_c_sources())
            .collect()
    }

    /// Collect extra embedded C include files from all extensions.
    pub fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.extra_c_include_files())
            .collect()
    }

    /// Collect extra CFLAGS from all extensions.
    pub fn extra_cflags(&self) -> Vec<String> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extension.extra_cflags())
            .collect()
    }
}

/// Rvr lifters and runtime hooks contributed by a VM configuration.
#[derive(Default)]
pub struct RvrExtensions {
    lifters: ExtensionRegistry,
    runtime_hooks: Vec<Box<dyn RvrRuntimeExtension>>,
}

impl RvrExtensions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lifters(&self) -> &ExtensionRegistry {
        &self.lifters
    }

    pub fn register_lifter(&mut self, ext: impl RvrExtension + 'static) {
        self.lifters.register(ext);
    }

    pub fn register_runtime_hook(&mut self, hook: impl RvrRuntimeExtension + 'static) {
        self.runtime_hooks.push(Box::new(hook));
    }

    pub fn into_runtime_hooks(self) -> Vec<Box<dyn RvrRuntimeExtension>> {
        self.runtime_hooks
    }
}

#[cfg(test)]
mod tests {
    use openvm_instructions::VmOpcode;
    use rvr_openvm_ir::{InstrAt, LiftedInstr};

    use super::*;
    use crate::opcode::NopInstr;

    struct ClaimingExtension;

    impl RvrExtension for ClaimingExtension {
        fn try_lift(&self, _insn: &RvrInstruction, pc: u64) -> Option<LiftedInstr> {
            Some(LiftedInstr::Body(InstrAt {
                pc,
                instr: Box::new(NopInstr),
                source_loc: None,
            }))
        }

        fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
            Vec::new()
        }
    }

    #[test]
    fn duplicate_opcode_claims_are_rejected() {
        let mut registry = ExtensionRegistry::new();
        registry.register(ClaimingExtension);
        registry.register(ClaimingExtension);
        let instruction =
            RvrInstruction::from_canonical(VmOpcode::from_usize(123), [0; 7], 2_013_265_921);

        assert!(matches!(
            registry.try_lift(&instruction, 0x100),
            Err(ExtensionError::DuplicateOpcodeClaim { .. })
        ));
    }
}
