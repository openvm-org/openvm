use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use rvr_openvm::{CProject, RvrExecutionKind};
use rvr_openvm_ir::{Block, CfgEffect, ExtEmitCtx, ExtInstr, InstrAt, Terminator};
use rvr_openvm_lift::ExtensionRegistry;

static NEXT_TEMP_DIR: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug)]
struct UnsupportedPreflightInstr;

impl ExtInstr for UnsupportedPreflightInstr {
    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

    fn opname(&self) -> &str {
        "unsupported"
    }

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Debug)]
struct PreflightCapableInstr;

impl ExtInstr for PreflightCapableInstr {
    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_preflight(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn ExtInstr> {
        Box::new(self.clone())
    }
}

fn block(instr: impl ExtInstr + 'static) -> Block {
    Block {
        start_pc: 0x100,
        end_pc: 0x108,
        instructions: vec![InstrAt {
            pc: 0x100,
            instr: Box::new(instr),
            source_loc: None,
        }],
        terminator: Terminator::Exit { code: 0 },
        terminator_pc: 0x104,
        terminator_source_loc: None,
    }
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(test_name: &str) -> Self {
        let id = NEXT_TEMP_DIR.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "rvr-openvm-{test_name}-{}-{id}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn preflight_compiler_rejects_unsupported_instruction() {
    let output = std::env::temp_dir().join("unused-preflight-capability-output");
    let project = CProject::new(&output, "test", RvrExecutionKind::Preflight);
    let error = project
        .write_all(
            &[block(UnsupportedPreflightInstr)],
            0x100,
            0x100,
            &ExtensionRegistry::new(),
        )
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "instruction unsupported at 0x100 does not support RVR preflight"
    );
}

#[test]
fn preflight_compiler_accepts_explicit_capability() {
    let output = TempDir::new("preflight-capable");
    let project = CProject::new(output.path(), "test", RvrExecutionKind::Preflight);

    project
        .write_all(
            &[block(PreflightCapableInstr)],
            0x100,
            0x100,
            &ExtensionRegistry::new(),
        )
        .unwrap();
}
