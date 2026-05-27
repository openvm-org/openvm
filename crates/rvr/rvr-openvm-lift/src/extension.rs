//! Extension registry for plugging in new opcode families.

use std::collections::HashMap;

use openvm_instructions::{instruction::Instruction, LocalOpcode, VmOpcode};
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::LiftedInstr;

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
    Ok(Some(AirIndex::new(*raw as u32)))
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
    #[error("failed to register host callbacks: {0}")]
    HostCallbackRegistration(String),
}

/// Trait for an rvr-openvm extension. Each extension handles a range of opcodes
/// and knows how to lift them to IR (with self-describing codegen via `ExtInstr::emit_c`).
pub trait RvrExtension<F: PrimeField32>: Send + Sync {
    /// Try to lift an OpenVM instruction into IR.
    /// Return `None` if this extension doesn't handle the opcode.
    /// Chip indices are stored on the extension and baked into IR nodes.
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr>;

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
    /// .so/.dylib. Default delegates to `staticlib_file()`.
    fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        vec![self.staticlib_file()]
    }

    /// A single embedded pre-built static library (.a) for this extension.
    fn staticlib_file(&self) -> (&'static str, &'static [u8]);

    /// Additional embedded C source files to compile alongside the generated
    /// code (e.g., precomputed tables).
    fn extra_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Additional embedded C files to write alongside the generated project
    /// because they are included by extension sources, but not compiled
    /// directly as translation units.
    fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        vec![]
    }

    /// Additional CFLAGS for compiling extension C sources (e.g., `-I`
    /// paths for submodule headers). Passed to the Makefile as EXT_CFLAGS.
    fn extra_cflags(&self) -> Vec<String> {
        vec![]
    }

    /// Register host-side callbacks for this extension with the loaded `.so`.
    /// Called after `register_openvm_callbacks` and before `rv_execute`.
    ///
    /// # Safety
    ///
    /// `lib` must be the rvr-compiled shared library this extension was lifted
    /// into.
    unsafe fn register_host_callbacks(
        &self,
        _lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        Ok(())
    }
}

/// Trait implemented by OpenVM extension owner types to contribute their rvr
/// lifting/codegen extensions during config assembly.
pub trait VmRvrExtension<F: PrimeField32> {
    fn extend_rvr(&self, _registry: &mut ExtensionRegistry<F>, _ctx: Option<&RvrExtensionCtx>) {}
}

impl<F: PrimeField32, EXT: VmRvrExtension<F>> VmRvrExtension<F> for Option<EXT> {
    fn extend_rvr(&self, registry: &mut ExtensionRegistry<F>, ctx: Option<&RvrExtensionCtx>) {
        if let Some(ext) = self {
            ext.extend_rvr(registry, ctx);
        }
    }
}

// ── Extension registry ───────────────────────────────────────────────────────

/// Registry of extensions, consulted during lifting and project generation.
pub struct ExtensionRegistry<F: PrimeField32> {
    extensions: Vec<Box<dyn RvrExtension<F>>>,
}

impl<F: PrimeField32> ExtensionRegistry<F> {
    /// Create an empty registry (no extensions).
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
        }
    }

    /// Register an extension.
    pub fn register(&mut self, ext: impl RvrExtension<F> + 'static) {
        self.extensions.push(Box::new(ext));
    }

    /// Try to lift an instruction through all registered extensions.
    /// Returns the first successful lift, or `None`.
    pub fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr> {
        self.extensions
            .iter()
            .find_map(|ext| ext.try_lift(insn, pc))
    }

    /// Collect all C headers from all registered extensions.
    pub fn c_headers(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.c_headers())
            .collect()
    }

    /// Collect all C source files from all registered extensions.
    pub fn c_sources(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.c_sources())
            .collect()
    }

    /// Collect all embedded static libraries for linking.
    pub fn staticlib_files(&self) -> Vec<(&'static str, &'static [u8])> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.staticlib_files())
            .collect()
    }

    /// Collect extra embedded C source files from all extensions.
    pub fn extra_c_sources(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extra_c_sources())
            .collect()
    }

    /// Collect extra embedded C include files from all extensions.
    pub fn extra_c_include_files(&self) -> Vec<(&'static str, &'static str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extra_c_include_files())
            .collect()
    }

    /// Collect extra CFLAGS from all extensions.
    pub fn extra_cflags(&self) -> Vec<String> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extra_cflags())
            .collect()
    }

    /// Whether any extensions are registered.
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }

    /// Register host callbacks for every extension in the registry.
    ///
    /// # Safety
    ///
    /// Same requirements as [`RvrExtension::register_host_callbacks`].
    pub unsafe fn register_host_callbacks(
        &self,
        lib: &libloading::Library,
    ) -> Result<(), ExtensionError> {
        for ext in &self.extensions {
            unsafe { ext.register_host_callbacks(lib)? };
        }
        Ok(())
    }
}

impl<F: PrimeField32> Default for ExtensionRegistry<F> {
    fn default() -> Self {
        Self::new()
    }
}
