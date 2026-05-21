use std::{
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use clap::{Parser, Subcommand};
use eyre::{bail, Context, Result};
use flate2::read::GzDecoder;
use indicatif::{ProgressBar, ProgressStyle};
use openvm_build::DEFAULT_RUSTUP_TOOLCHAIN_NAME;
use serde::Deserialize;
use tar::Archive;

use crate::get_target;

const RUSTC_FORK_REPO: &str = "openvm-org/rust";

const SUPPORTED_HOSTS: &[&str] = &[
    "x86_64-unknown-linux-gnu",
    "aarch64-unknown-linux-gnu",
    "aarch64-apple-darwin",
];

#[derive(Parser)]
#[command(
    name = "toolchain",
    about = "Manage the prebuilt openvm Rust toolchain"
)]
pub struct ToolchainCmd {
    #[command(subcommand)]
    pub action: ToolchainAction,
}

#[derive(Subcommand)]
pub enum ToolchainAction {
    /// Download and link the openvm Rust toolchain via `rustup toolchain link`.
    Install(InstallArgs),
    /// Remove an installed openvm Rust toolchain.
    Uninstall(UninstallArgs),
    /// List installed openvm Rust toolchains.
    List,
}

#[derive(Parser)]
pub struct InstallArgs {
    /// Release tag to install (e.g. `openvm-nightly-2026-05-13`).
    /// Defaults to the latest published release.
    #[arg(long)]
    pub version: Option<String>,
    /// Replace an existing link with the same name.
    #[arg(long)]
    pub force: bool,
}

#[derive(Parser)]
pub struct UninstallArgs {
    /// Toolchain to remove. Defaults to `DEFAULT_RUSTUP_TOOLCHAIN_NAME`.
    pub tag: Option<String>,
}

impl ToolchainCmd {
    pub fn run(&self) -> Result<()> {
        match &self.action {
            ToolchainAction::Install(args) => install(args),
            ToolchainAction::Uninstall(args) => uninstall(args),
            ToolchainAction::List => list(),
        }
    }
}

fn install(args: &InstallArgs) -> Result<()> {
    let host = host_triple()?;
    let tag = match args.version.as_deref() {
        Some(v) => v.to_string(),
        None => latest_release_tag()?,
    };

    let dest_dir = toolchains_root()?.join(&tag);
    if dest_dir.exists() && !args.force {
        println!(
            "openvm toolchain `{tag}` is already installed at {}.",
            dest_dir.display()
        );
        link_toolchain(&tag, &dest_dir)?;
        print_post_install_hint(&tag);
        return Ok(());
    }
    if dest_dir.exists() {
        fs::remove_dir_all(&dest_dir)
            .with_context(|| format!("failed to remove {}", dest_dir.display()))?;
    }

    let asset = format!("rust-toolchain-{host}.tar.gz");
    let url = format!("https://github.com/{RUSTC_FORK_REPO}/releases/download/{tag}/{asset}");
    println!("Installing openvm toolchain `{tag}` for host `{host}`");

    let tmp = tempfile::Builder::new()
        .prefix("openvm-toolchain-")
        .suffix(".tar.gz")
        .tempfile()
        .context("failed to create temp file")?;
    download_with_progress(&url, tmp.path())?;

    fs::create_dir_all(&dest_dir)
        .with_context(|| format!("failed to create {}", dest_dir.display()))?;
    extract_tarball(tmp.path(), &dest_dir)?;

    link_toolchain(&tag, &dest_dir)?;
    print_post_install_hint(&tag);
    Ok(())
}

fn uninstall(args: &UninstallArgs) -> Result<()> {
    let tag = args
        .tag
        .clone()
        .unwrap_or_else(|| DEFAULT_RUSTUP_TOOLCHAIN_NAME.to_string());

    let status = Command::new("rustup")
        .args(["toolchain", "uninstall", &tag])
        .status()
        .context("failed to run rustup toolchain uninstall")?;
    if !status.success() {
        bail!("rustup toolchain uninstall {tag} failed (exit {status})");
    }

    let dir = toolchains_root()?.join(&tag);
    if dir.exists() {
        fs::remove_dir_all(&dir).with_context(|| format!("failed to remove {}", dir.display()))?;
    }
    println!("Removed openvm toolchain `{tag}`.");
    Ok(())
}

fn list() -> Result<()> {
    let root = toolchains_root()?;
    if !root.exists() {
        println!("No openvm toolchains installed under {}.", root.display());
        return Ok(());
    }

    let mut entries: Vec<_> = fs::read_dir(&root)
        .with_context(|| format!("failed to read {}", root.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    if entries.is_empty() {
        println!("No openvm toolchains installed under {}.", root.display());
        return Ok(());
    }

    for entry in entries {
        let name = entry.file_name().to_string_lossy().to_string();
        let marker = if name == DEFAULT_RUSTUP_TOOLCHAIN_NAME {
            " (default)"
        } else {
            ""
        };
        println!("{name}{marker}");
    }
    Ok(())
}

fn host_triple() -> Result<String> {
    let host = get_target();
    if !SUPPORTED_HOSTS.contains(&host.as_str()) {
        bail!(
            "host triple `{host}` is not supported. Supported: {}",
            SUPPORTED_HOSTS.join(", ")
        );
    }
    Ok(host)
}

fn toolchains_root() -> Result<PathBuf> {
    let home =
        env::var_os("HOME").ok_or_else(|| eyre::eyre!("HOME environment variable is not set"))?;
    Ok(PathBuf::from(home).join(".openvm").join("toolchains"))
}

#[derive(Deserialize)]
struct Release {
    tag_name: String,
}

fn http_client() -> Result<reqwest::blocking::Client> {
    reqwest::blocking::Client::builder()
        .user_agent("cargo-openvm")
        .timeout(Duration::from_secs(60 * 30))
        .build()
        .context("failed to build HTTP client")
}

fn latest_release_tag() -> Result<String> {
    let url = format!("https://api.github.com/repos/{RUSTC_FORK_REPO}/releases/latest");
    let release: Release = http_client()?
        .get(&url)
        .header("Accept", "application/vnd.github+json")
        .send()
        .with_context(|| format!("failed to query {url}"))?
        .error_for_status()
        .with_context(|| format!("GitHub API request to {url} failed"))?
        .json()
        .context("failed to parse releases JSON")?;
    Ok(release.tag_name)
}

fn download_with_progress(url: &str, dest: &Path) -> Result<()> {
    let mut resp = http_client()?
        .get(url)
        .send()
        .with_context(|| format!("failed to GET {url}"))?
        .error_for_status()
        .with_context(|| format!("download of {url} failed"))?;

    let total = resp.content_length().unwrap_or(0);
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})",
        )
        .unwrap()
        .progress_chars("=>-"),
    );

    let mut out =
        fs::File::create(dest).with_context(|| format!("failed to create {}", dest.display()))?;
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = resp.read(&mut buf).context("read error from stream")?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).context("write error")?;
        pb.inc(n as u64);
    }
    pb.finish_and_clear();
    Ok(())
}

fn extract_tarball(archive: &Path, dest: &Path) -> Result<()> {
    let file =
        fs::File::open(archive).with_context(|| format!("failed to open {}", archive.display()))?;
    let gz = GzDecoder::new(io::BufReader::new(file));
    let mut ar = Archive::new(gz);
    ar.set_preserve_permissions(true);
    ar.unpack(dest)
        .with_context(|| format!("failed to unpack archive into {}", dest.display()))?;
    Ok(())
}

fn link_toolchain(tag: &str, dir: &Path) -> Result<()> {
    // If an `openvm-…` link already exists with the same target dir, leave it alone.
    // Otherwise uninstall first: `rustup toolchain link` refuses to overwrite.
    if !is_link_to(tag, dir) {
        let _ = Command::new("rustup")
            .args(["toolchain", "uninstall", tag])
            .status();
    }

    let status = Command::new("rustup")
        .args(["toolchain", "link", tag])
        .arg(dir)
        .status()
        .context("failed to run rustup toolchain link")?;
    if !status.success() {
        bail!(
            "rustup toolchain link {tag} {} failed (exit {status})",
            dir.display()
        );
    }
    Ok(())
}

/// Returns true if `rustup toolchain list --verbose` shows `tag` linked to `dir`.
/// Lets us skip a wasteful `rustup toolchain uninstall` + relink on idempotent installs.
fn is_link_to(tag: &str, dir: &Path) -> bool {
    let Ok(out) = Command::new("rustup")
        .args(["toolchain", "list", "--verbose"])
        .output()
    else {
        return false;
    };
    let stdout = String::from_utf8_lossy(&out.stdout);
    let dir_s = dir.to_string_lossy();
    stdout
        .lines()
        .filter_map(|l| {
            let (name, path) = l.split_once(char::is_whitespace)?;
            Some((name.trim(), path.trim()))
        })
        .any(|(name, path)| name == tag && path == dir_s)
}

fn print_post_install_hint(tag: &str) {
    println!();
    println!("Installed openvm toolchain `{tag}`.");
    println!();
    println!("Build guests with:");
    println!("    cargo openvm build");
    println!();
    println!("Or invoke the linked toolchain directly with atomic lowering:");
    println!("    RUSTFLAGS=\"-Cpasses=lower-atomic\" \\");
    println!("      cargo +{tag} build --target riscv64im-unknown-openvm-elf");
    if tag != DEFAULT_RUSTUP_TOOLCHAIN_NAME {
        println!();
        println!("Note: this cargo-openvm defaults to `{DEFAULT_RUSTUP_TOOLCHAIN_NAME}`.");
        println!("      Use `OPENVM_RUST_TOOLCHAIN={tag} cargo openvm build` to override.");
    }
}
