// Build script for the vendored Lean Swirl verifier (see README.md).
// Compiles the Lean-generated C sources under csrc/ into the
// `swirl_verify` and `swirl_dump_proof` executables using `leanc` from
// the pinned Lean toolchain. The dump tool is placed next to the verifier.

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Lean toolchain the vendored C sources were generated with. The
/// generated C is ABI-coupled to this runtime version; keep in sync with
/// README.md when refreshing the vendored sources.
const LEAN_TOOLCHAIN: &str = "leanprover/lean4:v4.26.0";

fn main() {
    println!("cargo:rerun-if-env-changed=ELAN_HOME");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed=csrc");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    ensure_pinned_leanc();

    let mut sources = Vec::new();
    collect_c_files(&csrc, &mut sources);
    assert!(!sources.is_empty(), "no C sources under {}", csrc.display());

    // Compile each Lean-generated C file. Flags mirror what lake used to
    // build the verifier in-repo (minus -DLEAN_EXPORTING / -fvisibility,
    // which only matter for shared-library builds).
    let objects: Vec<PathBuf> = std::thread::scope(|scope| {
        let jobs = std::thread::available_parallelism().map_or(4, |n| n.get());
        let mut handles = Vec::new();
        for chunk in sources.chunks(sources.len().div_ceil(jobs)) {
            let csrc = &csrc;
            let out_dir = &out_dir;
            handles.push(scope.spawn(move || {
                let mut objs = Vec::new();
                for src in chunk {
                    let rel = src.strip_prefix(csrc).unwrap();
                    let obj = out_dir.join(format!(
                        "{}.o",
                        rel.with_extension("").to_string_lossy().replace('/', "_")
                    ));
                    run(leanc(&[
                        "-c",
                        "-O3",
                        "-DNDEBUG",
                        "-fwrapv",
                        "-o",
                        obj.to_str().unwrap(),
                        src.to_str().unwrap(),
                    ]));
                    objs.push(obj);
                }
                objs
            }));
        }
        handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect()
    });

    let bin = out_dir.join("swirl_verify");
    let mut link_args: Vec<&str> = vec!["-o", bin.to_str().unwrap()];
    let obj_strs: Vec<String> = objects
        .iter()
        .map(|o| o.to_string_lossy().into_owned())
        .collect();
    link_args.extend(obj_strs.iter().map(String::as_str));
    run(leanc(&link_args));

    let dump_main = csrc.join("Tools/SwirlDumpProof.c");
    let dump_obj = out_dir.join("dump_Tools_SwirlDumpProof.o");
    run(leanc(&[
        "-c",
        "-O3",
        "-DNDEBUG",
        "-fwrapv",
        "-o",
        dump_obj.to_str().unwrap(),
        dump_main.to_str().unwrap(),
    ]));
    let raw_obj = out_dir.join("Swirl_Protocol_Noninteractive_Wire_Raw.o");
    assert!(
        raw_obj.exists(),
        "swirl_dump_proof needs {}",
        raw_obj.display()
    );
    let dump_bin = out_dir.join("swirl_dump_proof");
    run(leanc(&[
        "-o",
        dump_bin.to_str().unwrap(),
        dump_obj.to_str().unwrap(),
        raw_obj.to_str().unwrap(),
    ]));
}

fn leanc(args: &[&str]) -> Command {
    let mut cmd = Command::new("elan");
    cmd.args(["run", LEAN_TOOLCHAIN, "leanc"]);
    cmd.args(args);
    cmd
}

fn ensure_pinned_leanc() {
    let output = leanc(&["--version"]).output().unwrap_or_else(|e| {
        panic!(
            "building `openvm-certified-verifier` needs Lean toolchain \
             {LEAN_TOOLCHAIN}, but failed to run `elan`: {e}. Install elan \
             (https://github.com/leanprover/elan) and run \
             `elan toolchain install {LEAN_TOOLCHAIN}`."
        )
    });
    assert!(
        output.status.success(),
        "failed to run `leanc` from Lean toolchain {LEAN_TOOLCHAIN}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
}

fn collect_c_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_c_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "c")
            && path
                .file_name()
                .is_none_or(|name| name != "SwirlDumpProof.c")
        {
            out.push(path);
        }
    }
    out.sort();
}

fn run(mut cmd: Command) {
    let status = cmd
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {cmd:?}: {e}"));
    assert!(status.success(), "command failed ({status}): {cmd:?}");
}
