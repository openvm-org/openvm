//! Extension registry for plugging in new opcode families.

use std::path::{Path, PathBuf};

use openvm_instructions::instruction::Instruction;
use openvm_stark_backend::p3_field::PrimeField32;
use rvr_openvm_ir::LiftedInstr;

/// Trait for an rvr-openvm extension. Each extension handles a range of opcodes
/// and knows how to lift them to IR (with self-describing codegen via `ExtInstr::emit_c`).
pub trait RvrExtension<F: PrimeField32>: Send + Sync {
    /// Try to lift an OpenVM instruction into IR.
    /// Return `None` if this extension doesn't handle the opcode.
    /// Chip indices are stored on the extension and baked into IR nodes.
    fn try_lift(&self, insn: &Instruction<F>, pc: u32) -> Option<LiftedInstr>;

    /// C header files for this extension, as `(filename, content)` pairs.
    /// Written to the output directory and `#include`d in the generated code.
    fn c_headers(&self) -> Vec<(&str, &str)>;

    /// C source files for this extension, as `(filename, content)` pairs.
    /// Written to the output directory and compiled alongside the generated
    /// code by the Makefile (`$(wildcard *.c)`). This lets extension code
    /// call static inline tracer helpers directly instead of routing through
    /// separate Rust FFI wrappers.
    fn c_sources(&self) -> Vec<(&str, &str)> {
        vec![]
    }

    /// Paths to pre-built static libraries (.a) for this extension.
    /// These are linked into the final .so/.dylib.
    /// Default delegates to `staticlib_path()`.
    fn staticlib_paths(&self) -> Vec<&Path> {
        vec![self.staticlib_path()]
    }

    /// Path to a single pre-built static library (.a) for this extension.
    fn staticlib_path(&self) -> &Path;

    /// Additional C source file paths to compile alongside the generated
    /// code. Unlike `c_sources()` which provides inline content, these are
    /// paths to files on disk (e.g., precomputed tables in a submodule).
    fn extra_c_source_paths(&self) -> Vec<PathBuf> {
        vec![]
    }

    /// Additional CFLAGS for compiling extension C sources (e.g., `-I`
    /// paths for submodule headers). Passed to the Makefile as EXT_CFLAGS.
    fn extra_cflags(&self) -> Vec<String> {
        vec![]
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
    pub fn c_headers(&self) -> Vec<(&str, &str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.c_headers())
            .collect()
    }

    /// Collect all C source files from all registered extensions.
    pub fn c_sources(&self) -> Vec<(&str, &str)> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.c_sources())
            .collect()
    }

    /// Collect all staticlib paths for linking.
    pub fn staticlib_paths(&self) -> Vec<&Path> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.staticlib_paths())
            .collect()
    }

    /// Collect extra C source file paths from all extensions.
    pub fn extra_c_source_paths(&self) -> Vec<PathBuf> {
        self.extensions
            .iter()
            .flat_map(|ext| ext.extra_c_source_paths())
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
}

impl<F: PrimeField32> Default for ExtensionRegistry<F> {
    fn default() -> Self {
        Self::new()
    }
}
