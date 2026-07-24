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
struct LegacyPreflightOnlyInstr;

impl ExtInstr for LegacyPreflightOnlyInstr {
    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

    fn opname(&self) -> &str {
        "legacy-only"
    }

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

#[derive(Clone, Debug)]
struct CheckpointCapableInstr;

impl ExtInstr for CheckpointCapableInstr {
    fn emit_c(&self, _ctx: &mut dyn ExtEmitCtx) {}

    fn cfg_effect(&self) -> CfgEffect {
        CfgEffect::None
    }

    fn supports_checkpoint_preflight(&self) -> bool {
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
fn checkpoint_compiler_rejects_legacy_preflight_only_instruction() {
    let output = std::env::temp_dir().join("unused-checkpoint-capability-output");
    let project = CProject::new(&output, "test", RvrExecutionKind::CheckpointPreflight);
    let error = project
        .write_all(
            &[block(LegacyPreflightOnlyInstr)],
            0x100,
            0x100,
            &ExtensionRegistry::new(),
        )
        .unwrap_err();

    assert_eq!(
        error.to_string(),
        "instruction legacy-only at 0x100 does not support RVR preflight"
    );
}

#[test]
fn checkpoint_compiler_accepts_explicit_checkpoint_opt_in() {
    let output = TempDir::new("checkpoint-capable");
    let project = CProject::new(output.path(), "test", RvrExecutionKind::CheckpointPreflight);

    project
        .write_all(
            &[block(CheckpointCapableInstr)],
            0x100,
            0x100,
            &ExtensionRegistry::new(),
        )
        .unwrap();
}
