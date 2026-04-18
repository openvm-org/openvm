//! End-to-end validation for source annotations and DWARF line info.

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use eyre::{Context, ContextCompat, Result};
use openvm_instructions::exe::VmExe;
use openvm_platform::memory::MEM_SIZE;
use openvm_rv32im_transpiler::{
    Rv32ITranspilerExtension, Rv32IoTranspilerExtension, Rv32MTranspilerExtension,
};
use openvm_stark_sdk::p3_baby_bear::BabyBear;
use openvm_transpiler::{elf::Elf, transpiler::Transpiler, FromElf};
use rvr_openvm::{CompileOptions, GuestDebugMap, TracerMode};
use rvr_openvm_ir::SourceLoc;
use rvr_openvm_lift::ExtensionRegistry;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn openvm_root() -> PathBuf {
    workspace_root().join("openvm")
}

fn transpile(elf: Elf) -> Result<VmExe<BabyBear>> {
    Ok(VmExe::from_elf(
        elf,
        Transpiler::<BabyBear>::default()
            .with_extension(Rv32ITranspilerExtension)
            .with_extension(Rv32MTranspilerExtension)
            .with_extension(Rv32IoTranspilerExtension),
    )?)
}

fn find_paths_with_extension(dir: &Path, ext: &str) -> Result<Vec<PathBuf>> {
    Ok(fs::read_dir(dir)?
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.extension().and_then(|value| value.to_str()) == Some(ext))
        .collect())
}

fn llvm_dwarfdump_path() -> Option<PathBuf> {
    rvr_openvm::default_dwarfdump_cmd().map(PathBuf::from)
}

fn shared_lib_extension() -> &'static str {
    #[cfg(target_os = "macos")]
    {
        "dylib"
    }
    #[cfg(not(target_os = "macos"))]
    {
        "so"
    }
}

#[test]
fn test_debug_info_line_directives_and_dwarf() -> Result<()> {
    let elf_path = openvm_root().join("crates/toolchain/tests/tests/data/rv32im-fib-from-as");
    let data = fs::read(&elf_path)?;
    let exe = transpile(Elf::decode(&data, MEM_SIZE as u32)?)?;

    let tmp = tempfile::tempdir()?;
    let fake_source = tmp.path().join("synthetic_debug_source.rs");
    fs::write(&fake_source, "pub fn synthetic_debug_source() {}\n")?;

    let mut debug_map = GuestDebugMap::new();
    for (idx, (pc, _, _)) in exe
        .program
        .enumerate_by_pc()
        .into_iter()
        .take(5)
        .enumerate()
    {
        debug_map.insert(
            pc,
            SourceLoc::new(
                fake_source.to_str().context("non-utf8 temp source path")?,
                10 + idx as u32,
                "test_function",
            ),
        );
    }
    assert!(
        !debug_map.is_empty(),
        "synthetic debug map should not be empty"
    );

    let cache_dir = tmp.path().join("debug_test_cache");
    let registry = ExtensionRegistry::new();
    let opts = CompileOptions {
        base_name: Some("debug_test"),
        tracer_mode: TracerMode::Pure,
        extensions: &registry,
        chips: None,
        cache_dir: Some(&cache_dir),
        guest_debug_map: Some(&debug_map),
        native_debug_info: true,
    };

    let _compiled = rvr_openvm::compile_with_options(&exe, &opts)?;

    let mut found_line_directive = false;
    let mut found_comment = false;
    let fake_source_str = fake_source.to_str().unwrap();
    for entry in fs::read_dir(&cache_dir)? {
        let path = entry?.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("c") {
            let content = fs::read_to_string(&path)?;
            if content.contains("#line") && content.contains(fake_source_str) {
                found_line_directive = true;
            }
            if content.contains("// 0x") && content.contains("@ test_function") {
                found_comment = true;
            }
        }
    }

    assert!(
        found_line_directive,
        "expected #line directives in generated C"
    );
    assert!(found_comment, "expected source comments in generated C");

    let built_lib = find_paths_with_extension(&cache_dir, shared_lib_extension())?;
    assert!(
        !built_lib.is_empty(),
        "expected compiled shared library in cache dir"
    );

    if let Some(dwarfdump) = llvm_dwarfdump_path() {
        let object_files = find_paths_with_extension(&cache_dir, "o")?;
        assert!(
            !object_files.is_empty(),
            "expected generated object files for DWARF inspection"
        );

        let mut saw_synthetic_source = false;
        for object_file in object_files {
            let output = Command::new(&dwarfdump)
                .args([
                    "--debug-line",
                    "--show-sources",
                    object_file.to_str().unwrap(),
                ])
                .output()
                .context("failed to run llvm-dwarfdump")?;
            assert!(
                output.status.success(),
                "llvm-dwarfdump failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );

            let stdout = String::from_utf8_lossy(&output.stdout);
            if stdout.contains(fake_source_str) {
                saw_synthetic_source = true;
                break;
            }
        }

        assert!(
            saw_synthetic_source,
            "expected object-file DWARF line tables to reference synthetic source path"
        );
    } else {
        println!("skipping DWARF check: llvm-dwarfdump not found");
    }

    Ok(())
}
