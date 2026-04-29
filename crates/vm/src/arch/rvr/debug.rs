//! Guest debug info: maps OpenVM PCs to original source locations.
//!
//! A `GuestDebugMap` is built from the original RISC-V ELF (before
//! transpilation discards provenance) and threaded into the rvr-openvm
//! compilation pipeline to emit `#line` directives in generated C code.

use std::{
    collections::HashMap,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

use rvr_openvm_ir::SourceLoc;
use serde::{Deserialize, Serialize};

const ADDR2LINE_CHUNK_SIZE: usize = 1_000;

/// Maps OpenVM PCs to guest source locations.
///
/// Built externally (typically at ELF-transpile time) and passed into
/// the compilation pipeline as a sidecar.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GuestDebugMap {
    locations: HashMap<u32, SourceLoc>,
}

impl GuestDebugMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the source location for an OpenVM PC.
    pub fn get(&self, pc: u32) -> Option<&SourceLoc> {
        self.locations.get(&pc)
    }

    /// Insert a source location for a PC.
    pub fn insert(&mut self, pc: u32, loc: SourceLoc) {
        self.locations.insert(pc, loc);
    }

    pub fn len(&self) -> usize {
        self.locations.len()
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    /// Build a debug map from an ELF file using `llvm-addr2line`.
    ///
    /// `elf_path` — path to the RISC-V ELF with DWARF debug info.
    /// `pcs` — the set of OpenVM PCs to resolve (typically all instruction PCs).
    /// `addr2line_cmd` — the llvm-addr2line binary (e.g. "llvm-addr2line" or "llvm-addr2line-20").
    ///
    /// For standard RV32IM transpilation, OpenVM PCs equal ELF PCs, so
    /// this can be called directly with the PCs from the lifted IR.
    pub fn from_elf(elf_path: &Path, pcs: &[u32], addr2line_cmd: &str) -> Result<Self, String> {
        if pcs.is_empty() {
            return Ok(Self::new());
        }

        let started_at = Instant::now();
        let mut map = Self::new();
        let mut resolved_total = 0usize;
        let mut last_progress_at = started_at;
        let mut last_chunk_log_at = started_at;
        let total_chunks = pcs.len().div_ceil(ADDR2LINE_CHUNK_SIZE);

        for (chunk_idx, chunk) in pcs.chunks(ADDR2LINE_CHUNK_SIZE).enumerate() {
            let chunk_num = chunk_idx + 1;
            if total_chunks > 1
                && (chunk_num <= 3
                    || chunk_num == total_chunks
                    || last_chunk_log_at.elapsed() >= Duration::from_secs(10))
            {
                eprintln!(
                    "[rvr-openvm] Guest debug info chunk {}/{} ({} PCs resolved so far)",
                    chunk_num, total_chunks, resolved_total
                );
                last_chunk_log_at = Instant::now();
            }
            let mut child = Command::new(addr2line_cmd)
                .arg("-e")
                .arg(elf_path)
                .arg("-f")
                .arg("-C")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| format!("failed to run {addr2line_cmd}: {e}"))?;

            {
                let mut stdin = child
                    .stdin
                    .take()
                    .ok_or_else(|| format!("failed to open stdin for {addr2line_cmd}"))?;
                for pc in chunk {
                    writeln!(stdin, "0x{pc:x}")
                        .map_err(|e| format!("failed to write addr2line input: {e}"))?;
                }
            }

            let output = child
                .wait_with_output()
                .map_err(|e| format!("failed to wait for {addr2line_cmd}: {e}"))?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(format!("{addr2line_cmd} failed: {stderr}"));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_addr2line_chunk(&mut map, chunk, &stdout, pcs.len(), resolved_total);
            resolved_total += chunk.len();

            if pcs.len() > ADDR2LINE_CHUNK_SIZE
                && started_at.elapsed() >= Duration::from_secs(10)
                && last_progress_at.elapsed() >= Duration::from_secs(5)
            {
                eprintln!(
                    "[rvr-openvm] Guest debug info progress: {resolved_total}/{} PCs ({:.0}s elapsed)",
                    pcs.len(),
                    started_at.elapsed().as_secs_f64()
                );
                last_progress_at = Instant::now();
            }
        }

        Ok(map)
    }
}

fn parse_addr2line_chunk(
    map: &mut GuestDebugMap,
    pcs: &[u32],
    stdout: &str,
    total_pcs: usize,
    resolved_offset: usize,
) {
    let mut lines = stdout.lines();
    for (resolved, &pc) in pcs.iter().enumerate() {
        let Some(func_line) = lines.next() else {
            eprintln!(
                "warn: addr2line output truncated after {} of {} PCs",
                resolved_offset + resolved,
                total_pcs
            );
            break;
        };
        let Some(loc_line) = lines.next() else {
            eprintln!(
                "warn: addr2line output truncated after {} of {} PCs",
                resolved_offset + resolved,
                total_pcs
            );
            break;
        };

        let loc = parse_addr2line_location(func_line.trim(), loc_line.trim());
        if loc.is_valid() {
            map.locations.insert(pc, loc);
        }
    }
}

/// Best-effort `llvm-addr2line` command lookup.
#[must_use]
pub fn default_addr2line_cmd() -> String {
    rvr_openvm::default_addr2line_cmd()
}

/// Parse a single addr2line function/location pair into a `SourceLoc`.
fn parse_addr2line_location(func_line: &str, loc_line: &str) -> SourceLoc {
    let function = if func_line == "??" {
        String::new()
    } else {
        func_line.to_string()
    };

    let (file, line) = loc_line.rfind(':').map_or_else(
        || (String::from("??"), 0),
        |colon_idx| {
            let file = &loc_line[..colon_idx];
            let line_part = &loc_line[colon_idx + 1..];
            let line_str = line_part.split_whitespace().next().unwrap_or("0");
            let line = line_str.parse::<u32>().unwrap_or(0);
            (file.to_string(), line)
        },
    );

    SourceLoc::new(&file, line, &function)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_addr2line_location_valid() {
        let loc = parse_addr2line_location("main", "/path/to/file.rs:42");
        assert!(loc.is_valid());
        assert_eq!(loc.file, "/path/to/file.rs");
        assert_eq!(loc.line, 42);
        assert_eq!(loc.function, "main");
    }

    #[test]
    fn test_parse_addr2line_location_with_discriminator() {
        let loc = parse_addr2line_location("foo", "/path/file.rs:10 (discriminator 1)");
        assert!(loc.is_valid());
        assert_eq!(loc.line, 10);
    }

    #[test]
    fn test_parse_addr2line_location_unknown() {
        let loc = parse_addr2line_location("??", "??:0");
        assert!(!loc.is_valid());
    }
}
