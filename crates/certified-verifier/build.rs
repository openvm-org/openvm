// Build script for the vendored Lean certified verifiers (see README.md).
// Compiles the Lean-generated C sources under csrc/ into a static library
// linked into this crate, using `leanc` from the pinned Lean toolchain.
// `vm_dump_proof` is an executable, as it is a wire-format test utility.

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
    println!("cargo:rerun-if-changed=src/ffi");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest_dir.join("csrc");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    ensure_pinned_leanc();
    let lean_prefix = lean_prefix();

    let mut sources = Vec::new();
    collect_c_files(&csrc, &mut sources);
    assert!(!sources.is_empty(), "no C sources under {}", csrc.display());

    // Compile the verifier's Lean-generated C link closure, excluding the
    // executable entry point. Flags mirror what lake used to build the
    // verifier in-repo (minus -DLEAN_EXPORTING / -fvisibility, which only
    // matter for shared-library builds).
    sources.retain(|path| {
        let relative_path = path.strip_prefix(&csrc).unwrap();
        relative_path != Path::new("VmVerifier/Main.c")
            && relative_path != Path::new("VmVerifier/DumpProof.c")
    });
    let mut objects: Vec<PathBuf> = std::thread::scope(|scope| {
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

    let ffi_src = manifest_dir.join("src/ffi/vm_verify.c");
    let ffi_obj = out_dir.join("openvm_verify_ffi.o");
    run(leanc(&[
        "-c",
        "-O3",
        "-DNDEBUG",
        "-fwrapv",
        "-o",
        ffi_obj.to_str().unwrap(),
        ffi_src.to_str().unwrap(),
    ]));
    objects.push(ffi_obj);

    let verifier_lib = out_dir.join("libopenvm_certified_verifier.a");
    let mut archive = Command::new(lean_prefix.join("bin/llvm-ar"));
    archive.args(["crs", verifier_lib.to_str().unwrap()]);
    archive.args(&objects);
    run(archive);

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=openvm_certified_verifier");
    emit_leanc_link_flags(&lean_prefix);

    let dump_main = csrc.join("VmVerifier/DumpProof.c");
    let dump_obj = out_dir.join("dump_VmVerifier_DumpProof.o");
    run(leanc(&[
        "-c",
        "-O3",
        "-DNDEBUG",
        "-fwrapv",
        "-o",
        dump_obj.to_str().unwrap(),
        dump_main.to_str().unwrap(),
    ]));
    let dump_bin = out_dir.join("vm_dump_proof");
    run(leanc(&[
        "-o",
        dump_bin.to_str().unwrap(),
        dump_obj.to_str().unwrap(),
        verifier_lib.to_str().unwrap(),
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

fn lean_prefix() -> PathBuf {
    let output = Command::new("elan")
        .args(["run", LEAN_TOOLCHAIN, "lean", "--print-prefix"])
        .output()
        .unwrap_or_else(|e| panic!("failed to query Lean toolchain prefix: {e}"));
    assert!(
        output.status.success(),
        "failed to query Lean toolchain prefix: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    PathBuf::from(String::from_utf8(output.stdout).unwrap().trim())
}

/// Forward the pinned toolchain's static Lean link recipe to Cargo.
///
/// `leanc --print-ldflags` accounts for platform-specific runtime choices
/// such as libstdc++ versus libc++. Cargo needs the library kind separately,
/// so prefer a static archive when the Lean toolchain supplies one and fall
/// back to the platform's dynamic library otherwise.
fn emit_leanc_link_flags(lean_prefix: &Path) {
    let output = leanc(&["--print-ldflags"])
        .output()
        .unwrap_or_else(|e| panic!("failed to query leanc linker flags: {e}"));
    assert!(
        output.status.success(),
        "failed to query leanc linker flags: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let output = String::from_utf8(output.stdout).expect("leanc linker flags must be UTF-8");
    let search_paths = [lean_prefix.join("lib/lean"), lean_prefix.join("lib")];
    for path in &search_paths {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    let libraries: Vec<_> = output
        .split_whitespace()
        .filter_map(|flag| flag.strip_prefix("-l").filter(|lib| !lib.is_empty()))
        .collect();
    assert!(
        libraries.contains(&"leanrt"),
        "leanc linker flags did not include the Lean runtime: {output}"
    );

    for flag in output.split_whitespace() {
        if flag == "-pthread" {
            println!("cargo:rustc-link-lib=dylib=pthread");
        } else if let Some(lib) = flag.strip_prefix("-l").filter(|lib| !lib.is_empty()) {
            assert!(
                !lib.starts_with(':'),
                "unsupported verbatim library in leanc linker flags: {flag}"
            );
            let kind = if search_paths
                .iter()
                .any(|path| path.join(format!("lib{lib}.a")).is_file())
            {
                "static"
            } else {
                assert!(
                    !search_paths.iter().any(|path| {
                        path.join(format!("lib{lib}.so")).is_file()
                            || path.join(format!("lib{lib}.dylib")).is_file()
                            || path.join(format!("lib{lib}.dll.a")).is_file()
                    }),
                    "refusing to dynamically link {lib} from the Lean toolchain"
                );
                "dylib"
            };
            println!("cargo:rustc-link-lib={kind}={lib}");
        }
    }
}

fn collect_c_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            collect_c_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "c") {
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
