use std::{
    env,
    ffi::OsStr,
    fmt::Write as _,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use rvr_openvm::{RuntimeToolchain, RvrExecutionKind};
use sha2::{Digest, Sha256};

const CACHE_ENV: &str = "OPENVM_RVR_NATIVE_CACHE_DIR";
const ARTIFACT_DIGEST: &str = "artifact.sha256";

const BUILD_ENV: &[&str] = &[
    "PATH",
    "OPT",
    "DEBUG",
    "SANITIZERS",
    "LTO",
    "LDFLAGS",
    "LDLIBS",
    "CPATH",
    "C_INCLUDE_PATH",
    "LIBRARY_PATH",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "SYSROOT",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
];

pub(super) struct NativeArtifactCache {
    root: PathBuf,
}

impl NativeArtifactCache {
    pub(super) fn configured(kind: RvrExecutionKind, native_debug_info: bool) -> Option<Self> {
        if !matches!(
            kind,
            RvrExecutionKind::Metered | RvrExecutionKind::Preflight
        ) || native_debug_info
        {
            return None;
        }
        let root = env::var_os(CACHE_ENV)?;
        if root.is_empty() {
            return None;
        }
        Some(Self {
            root: PathBuf::from(root),
        })
    }

    pub(super) fn project_key(
        &self,
        project_dir: &Path,
        make_args: &[String],
        kind: RvrExecutionKind,
        toolchain: &RuntimeToolchain,
    ) -> Result<String, String> {
        let mut hasher = Sha256::new();
        update(&mut hasher, b"openvm-rvr-native-cache");
        update(&mut hasher, format!("{kind:?}").as_bytes());
        update(&mut hasher, env::consts::OS.as_bytes());
        update(&mut hasher, env::consts::ARCH.as_bytes());

        for path in project_files(project_dir)? {
            let relative = path
                .strip_prefix(project_dir)
                .map_err(|error| format!("invalid generated-project path: {error}"))?;
            update(&mut hasher, normalized_path(relative).as_bytes());
            let mut file = File::open(&path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let length = file
                .metadata()
                .map_err(|error| format!("failed to stat {}: {error}", path.display()))?
                .len();
            hasher.update(length.to_le_bytes());
            let mut buffer = [0u8; 1024 * 1024];
            loop {
                let read = file
                    .read(&mut buffer)
                    .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
                if read == 0 {
                    break;
                }
                hasher.update(&buffer[..read]);
            }
        }

        let project = project_dir.to_string_lossy();
        for arg in make_args {
            update(
                &mut hasher,
                arg.replace(project.as_ref(), "$PROJECT").as_bytes(),
            );
        }
        for value in [
            toolchain.compiler.as_str(),
            toolchain.linker.as_str(),
            toolchain.make.as_str(),
            toolchain.host_os,
        ] {
            update(&mut hasher, value.as_bytes());
        }
        command_identity(&mut hasher, &toolchain.compiler, &["--version"])?;
        command_identity(
            &mut hasher,
            &toolchain.compiler,
            &[
                "-march=native",
                "-###",
                "-c",
                "-x",
                "c",
                "/dev/null",
                "-o",
                "/dev/null",
            ],
        )?;
        command_identity(
            &mut hasher,
            resolved_linker_for_fingerprint(&toolchain.linker),
            &["--version"],
        )?;
        command_identity(&mut hasher, &toolchain.make, &["--version"])?;

        for name in BUILD_ENV {
            update(&mut hasher, name.as_bytes());
            update_env(&mut hasher, env::var_os(name).as_deref());
        }
        Ok(hex(hasher.finalize().as_slice()))
    }

    pub(super) fn lookup(&self, key: &str, library_name: &str) -> Result<Option<PathBuf>, String> {
        let entry = self.entry(key);
        if !entry.exists() {
            return Ok(None);
        }
        let library = entry.join(library_name);
        let digest_path = entry.join(ARTIFACT_DIGEST);
        if !is_regular_file(&library) || !is_regular_file(&digest_path) {
            self.quarantine(&entry);
            return Ok(None);
        }
        let expected = fs::read(&digest_path)
            .map_err(|error| format!("failed to read {}: {error}", digest_path.display()))?;
        if expected.len() != 32 || artifact_digest(&library)? != expected.as_slice() {
            self.quarantine(&entry);
            return Ok(None);
        }
        Ok(Some(library))
    }

    pub(super) fn publish(
        &self,
        key: &str,
        library_name: &str,
        source: &Path,
    ) -> Result<(), String> {
        let entry = self.entry(key);
        if entry.exists() {
            return Ok(());
        }
        let parent = entry
            .parent()
            .ok_or_else(|| "native cache entry has no parent".to_string())?;
        fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
        let staging = tempfile::Builder::new()
            .prefix(".openvm-rvr-cache-")
            .tempdir_in(parent)
            .map_err(|error| format!("failed to stage native cache entry: {error}"))?;
        let staged_library = staging.path().join(library_name);
        fs::copy(source, &staged_library)
            .map_err(|error| format!("failed to stage {}: {error}", source.display()))?;
        File::open(&staged_library)
            .and_then(|file| file.sync_all())
            .map_err(|error| format!("failed to sync {}: {error}", staged_library.display()))?;
        let digest = artifact_digest(&staged_library)?;
        let digest_path = staging.path().join(ARTIFACT_DIGEST);
        let mut digest_file = File::create(&digest_path)
            .map_err(|error| format!("failed to create {}: {error}", digest_path.display()))?;
        digest_file
            .write_all(&digest)
            .and_then(|()| digest_file.sync_all())
            .map_err(|error| format!("failed to write {}: {error}", digest_path.display()))?;
        match fs::rename(staging.path(), &entry) {
            Ok(()) => Ok(()),
            Err(_) if entry.exists() => Ok(()),
            Err(error) => Err(format!(
                "failed to publish native cache entry {}: {error}",
                entry.display()
            )),
        }
    }

    pub(super) fn quarantine_key(&self, key: &str) {
        self.quarantine(&self.entry(key));
    }

    fn entry(&self, key: &str) -> PathBuf {
        self.root.join(&key[..2]).join(key)
    }

    fn quarantine(&self, entry: &Path) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let Some(name) = entry.file_name() else {
            return;
        };
        let invalid = entry.with_file_name(format!(
            ".invalid-{}-{nonce}-{}",
            std::process::id(),
            name.to_string_lossy()
        ));
        if let Err(error) = fs::rename(entry, &invalid) {
            tracing::warn!(
                path = %entry.display(),
                %error,
                "failed to quarantine invalid RVR native cache entry"
            );
        }
    }
}

fn resolved_linker_for_fingerprint(linker: &str) -> &str {
    let linker_works = Command::new(linker)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success());
    if linker == "lld" && !linker_works {
        "ld.lld"
    } else {
        linker
    }
}

fn project_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    fn visit(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
        let mut entries = fs::read_dir(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let file_type = entry
                .file_type()
                .map_err(|error| format!("failed to stat {}: {error}", path.display()))?;
            if file_type.is_dir() {
                visit(&path, files)?;
            } else if file_type.is_file() {
                files.push(path);
            } else {
                return Err(format!(
                    "generated project contains unsupported path {}",
                    path.display()
                ));
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    visit(root, &mut files)?;
    Ok(files)
}

fn normalized_path(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

fn command_identity(hasher: &mut Sha256, command: &str, args: &[&str]) -> Result<(), String> {
    let output = Command::new(command)
        .args(args)
        .output()
        .map_err(|error| format!("failed to fingerprint {command}: {error}"))?;
    update(hasher, command.as_bytes());
    for arg in args {
        update(hasher, arg.as_bytes());
    }
    hasher.update(output.status.code().unwrap_or(-1).to_le_bytes());
    update(hasher, &output.stdout);
    update(hasher, &output.stderr);
    Ok(())
}

fn artifact_digest(path: &Path) -> Result<[u8; 32], String> {
    let mut file =
        File::open(path).map_err(|error| format!("failed to read {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finalize().into())
}

fn is_regular_file(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|metadata| metadata.file_type().is_file())
}

fn update(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn update_env(hasher: &mut Sha256, value: Option<&OsStr>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            update(hasher, value.to_string_lossy().as_bytes());
        }
        None => hasher.update([0]),
    }
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut output, "{byte:02x}").unwrap();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    const LIBRARY: &str = "libopenvm.so";

    #[test]
    fn published_artifact_round_trips() {
        let root = tempfile::tempdir().unwrap();
        let source_dir = tempfile::tempdir().unwrap();
        let source = source_dir.path().join(LIBRARY);
        fs::write(&source, b"native artifact").unwrap();
        let cache = NativeArtifactCache {
            root: root.path().to_path_buf(),
        };

        cache.publish(KEY, LIBRARY, &source).unwrap();
        let cached = cache.lookup(KEY, LIBRARY).unwrap().unwrap();

        assert_eq!(fs::read(cached).unwrap(), b"native artifact");
    }

    #[test]
    fn corrupted_artifact_is_rejected() {
        let root = tempfile::tempdir().unwrap();
        let source_dir = tempfile::tempdir().unwrap();
        let source = source_dir.path().join(LIBRARY);
        fs::write(&source, b"native artifact").unwrap();
        let cache = NativeArtifactCache {
            root: root.path().to_path_buf(),
        };

        cache.publish(KEY, LIBRARY, &source).unwrap();
        fs::write(cache.entry(KEY).join(LIBRARY), b"corrupted").unwrap();

        assert_eq!(cache.lookup(KEY, LIBRARY).unwrap(), None);
        assert!(!cache.entry(KEY).exists());
    }

    #[test]
    fn unset_and_empty_environment_values_have_distinct_keys() {
        let mut unset = Sha256::new();
        update_env(&mut unset, None);
        let mut empty = Sha256::new();
        update_env(&mut empty, Some(OsStr::new("")));

        assert_ne!(unset.finalize(), empty.finalize());
    }
}
