use std::{
    collections::HashMap,
    fs,
    path::{Component, Path, PathBuf},
    time::{Duration, SystemTime},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use eyre::{bail, eyre, Context, Result};
use flate2::{write::GzEncoder, Compression};
use fxprof_processed_profile::{
    Category, CategoryColor, CpuDelta, FrameAddress, FrameFlags, FrameSymbolInfo, LibraryInfo,
    Profile, SamplingInterval, SourceLocation, SubcategoryHandle, Timestamp,
};
use object::{Object, ObjectSymbol, SymbolKind};
use openvm_circuit::arch::rvr::{GuestProfileConfig, RawGuestProfile, RAW_GUEST_PROFILE_VERSION};
use reqwest::{blocking::Client, header::ACCEPT};
use samply_debugid::DebugIdExt;
use serde_json::Value;

const DEFAULT_UPLOAD_URL: &str = "https://api.profiler.firefox.com/compressed-store";
const FIREFOX_ACCEPT: &str = "application/vnd.firefox-profiler+json;version=1.0";

/// Coordinates one guest execution capture and its Firefox profile conversion.
pub struct FirefoxProfiler {
    guest_elf_path: PathBuf,
    config: GuestProfileConfig,
    _raw_profile: tempfile::NamedTempFile,
}

impl FirefoxProfiler {
    pub fn new(guest_elf_path: impl Into<PathBuf>, sample_hz: u32) -> Result<Self> {
        let raw_profile =
            tempfile::NamedTempFile::new().context("failed to create temporary RVR profile")?;
        let config =
            GuestProfileConfig::raw(raw_profile.path(), sample_hz).map_err(|error| eyre!(error))?;
        Ok(Self {
            guest_elf_path: guest_elf_path.into(),
            config,
            _raw_profile: raw_profile,
        })
    }

    /// Explicit VM/SDK configuration for the execution being profiled.
    pub fn config(&self) -> &GuestProfileConfig {
        &self.config
    }

    /// Convert the captured raw samples into a reusable Firefox artifact.
    pub fn finish(self) -> Result<FirefoxProfile> {
        FirefoxProfile::from_raw_guest_stacks(
            self.config.output(),
            &self.guest_elf_path,
            None,
            self.config.sample_hz(),
        )
    }
}

/// A generated, symbolicated Firefox Profiler artifact.
pub struct FirefoxProfile {
    compressed: Vec<u8>,
    sample_count: usize,
}

impl FirefoxProfile {
    /// Convert an ordered v3 raw guest profile into a symbolicated Firefox profile.
    ///
    /// `native_artifact_path` is retained for source compatibility. If supplied,
    /// it overrides the generated module path recorded in the raw profile.
    pub fn from_raw_guest_stacks(
        raw_profile_path: &Path,
        guest_elf_path: &Path,
        native_artifact_path: Option<&Path>,
        _sample_hz: u32,
    ) -> Result<Self> {
        let raw = parse_raw_profile(raw_profile_path)?;
        let profile = build_firefox_profile(&raw, guest_elf_path, native_artifact_path)?;
        Ok(Self {
            compressed: compress_profile(&profile)?,
            sample_count: raw.samples.len(),
        })
    }

    /// Number of ordered guest call-stack samples in this profile.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// The gzip-compressed Firefox Profiler JSON payload.
    pub fn compressed(&self) -> &[u8] {
        &self.compressed
    }

    /// Save the gzip-compressed profile to `path`.
    pub fn save(&self, path: &Path) -> Result<()> {
        fs::write(path, &self.compressed)
            .with_context(|| format!("failed to write Firefox profile to {}", path.display()))
    }

    /// Upload the profile using Firefox Profiler's compressed-store protocol.
    pub fn upload(&self) -> Result<String> {
        upload_profile(&self.compressed)
    }
}

fn parse_raw_profile(path: &Path) -> Result<RawGuestProfile> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read RVR samples from {}", path.display()))?;
    let raw: RawGuestProfile =
        serde_json::from_str(&contents).context("failed to parse versioned RVR profile")?;
    if raw.version != RAW_GUEST_PROFILE_VERSION {
        bail!(
            "unsupported RVR profile version {}; expected {}",
            raw.version,
            RAW_GUEST_PROFILE_VERSION
        );
    }
    if raw.samples.is_empty() {
        bail!(
            "RVR execution completed before any profile samples were captured; try a larger workload or a higher --rate"
        );
    }
    if raw.dropped_samples != 0
        || raw.timer_overruns != 0
        || raw.timer_arm_failures != 0
        || raw.clock_failures != 0
    {
        bail!(
            "RVR profile is incomplete (dropped={}, overruns={}, arm_failures={}, clock_failures={}); capture again at a sustainable sampling rate",
            raw.dropped_samples,
            raw.timer_overruns,
            raw.timer_arm_failures,
            raw.clock_failures
        );
    }
    if raw.delivered_samples != raw.samples.len() as u64 {
        bail!(
            "RVR profile sample count does not match delivery metadata (retained={}, delivered={})",
            raw.samples.len(),
            raw.delivered_samples
        );
    }
    if raw.samples.iter().any(|sample| sample.stack_truncated) {
        bail!(
            "RVR profile contains a truncated guest stack; capture again with a larger stack limit"
        );
    }
    Ok(raw)
}

fn build_firefox_profile(
    raw: &RawGuestProfile,
    guest_elf_path: &Path,
    native_artifact_path: Option<&Path>,
) -> Result<Profile> {
    let guest_resolver = BinaryResolver::new(guest_elf_path)?;
    let observed_interval_ns = observed_interval_ns(raw);
    let interval = SamplingInterval::from_hz((1_000_000_000.0 / observed_interval_ns) as f32);
    let start_time = SystemTime::UNIX_EPOCH
        .checked_add(Duration::from_nanos(raw.start_unix_time_ns))
        .unwrap_or(SystemTime::UNIX_EPOCH);
    let mut profile = Profile::new("OpenVM guest execution", start_time.into(), interval);
    profile.set_os_name("OpenVM RV64 guest");
    profile.set_symbolicated(true);
    let zero = Timestamp::from_millis_since_reference(0.0);
    let process = profile.add_process("OpenVM guest", 1, zero);
    let thread = profile.add_thread(process, raw.owner_tid.max(1) as u32, zero, true);
    profile.set_thread_name(thread, "RV64 guest execution");
    profile.add_initial_selected_thread(thread);

    let guest_lib = add_library(&mut profile, guest_elf_path, None, "riscv64")?;
    let mut native_modules = Vec::with_capacity(raw.native_modules.len());
    for module in &raw.native_modules {
        let path = if module.generated {
            native_artifact_path.unwrap_or_else(|| Path::new(&module.path))
        } else {
            Path::new(&module.path)
        };
        match BinaryResolver::new(path) {
            Ok(resolver) => {
                let library = add_library(&mut profile, path, Some(&module.name), "x86_64")?;
                native_modules.push(NativeModule {
                    resolver: Some(resolver),
                    library,
                    generated: module.generated,
                });
            }
            Err(error) if module.generated => {
                return Err(error).with_context(|| {
                    format!(
                        "failed to open generated native module {} recorded in the raw profile",
                        path.display()
                    )
                });
            }
            Err(_) => {
                native_modules.push(NativeModule {
                    resolver: None,
                    library: add_unresolved_library(&mut profile, &module.name, "x86_64"),
                    generated: false,
                });
            }
        }
    }
    let unknown_native_lib = profile.add_lib(LibraryInfo {
        name: "[unknown native]".to_string(),
        debug_name: "[unknown native]".to_string(),
        path: "[unknown native]".to_string(),
        debug_path: "[unknown native]".to_string(),
        debug_id: debugid::DebugId::nil(),
        code_id: None,
        arch: Some("x86_64".to_string()),
    });

    let guest_category: SubcategoryHandle = profile
        .handle_for_category(Category("Guest", CategoryColor::Yellow))
        .into();
    let mut native_symbols = HashMap::new();
    let mut strings = HashMap::new();
    let mut previous_cpu_time = raw.start_cpu_time_ns;

    for sample in &raw.samples {
        let timestamp_ns = sample.wall_time_ns.saturating_sub(raw.start_wall_time_ns);
        let timestamp = Timestamp::from_millis_since_reference(timestamp_ns as f64 / 1_000_000.0);
        let cpu_delta = CpuDelta::from_nanos(sample.cpu_time_ns.saturating_sub(previous_cpu_time));
        previous_cpu_time = sample.cpu_time_ns;
        let mut frame_handles = Vec::new();

        for &return_pc in &sample.guest_return_pcs {
            let lookup_pc = return_pc.saturating_sub(1);
            emit_resolved_pc(
                &mut profile,
                &mut native_symbols,
                &mut strings,
                &mut frame_handles,
                guest_lib,
                0,
                lookup_pc,
                guest_resolver.resolve(lookup_pc),
                guest_category,
                true,
                "guest",
            );
        }
        let native_leaf_is_guest_execution = sample.native_leaf.as_ref().is_some_and(|leaf| {
            leaf.module_index
                .and_then(|index| native_modules.get(index as usize))
                .is_some_and(|module| module.is_guest_execution_frame(leaf.pc))
        });
        if !native_leaf_is_guest_execution {
            if let Some(callsite_pc) = sample.guest_callsite_pc {
                emit_resolved_pc(
                    &mut profile,
                    &mut native_symbols,
                    &mut strings,
                    &mut frame_handles,
                    guest_lib,
                    0,
                    callsite_pc,
                    guest_resolver.resolve(callsite_pc),
                    guest_category,
                    false,
                    "guest",
                );
            }
        }

        if let Some(native_leaf) = &sample.native_leaf {
            emit_native_leaf(
                &mut profile,
                &mut native_symbols,
                &mut strings,
                &mut frame_handles,
                &native_modules,
                unknown_native_lib,
                &guest_resolver,
                native_leaf.module_index,
                native_leaf.pc,
                guest_category,
            );
        }

        let mut stack = None;
        for frame in frame_handles {
            stack = Some(profile.handle_for_stack(frame, stack));
        }
        profile.add_sample(thread, timestamp, stack, cpu_delta, 1);
    }

    let end_ns = raw.end_wall_time_ns.saturating_sub(raw.start_wall_time_ns);
    let end = Timestamp::from_millis_since_reference(end_ns as f64 / 1_000_000.0);
    profile.set_process_end_time(process, end);
    profile.set_thread_end_time(thread, end);
    Ok(profile)
}

#[allow(clippy::too_many_arguments)]
fn emit_native_leaf(
    profile: &mut Profile,
    native_symbols: &mut HashMap<(usize, u32), fxprof_processed_profile::NativeSymbolHandle>,
    strings: &mut HashMap<String, fxprof_processed_profile::StringHandle>,
    output: &mut Vec<fxprof_processed_profile::FrameHandle>,
    modules: &[NativeModule],
    unknown_lib: fxprof_processed_profile::LibraryHandle,
    guest_resolver: &BinaryResolver,
    module_index: Option<u32>,
    native_pc: u64,
    category: SubcategoryHandle,
) {
    let Some(module) = module_index.and_then(|index| modules.get(index as usize)) else {
        let pc = u32::try_from(native_pc).unwrap_or(u32::MAX);
        emit_frame_chain(
            profile,
            native_symbols,
            strings,
            output,
            unknown_lib,
            usize::MAX,
            pc,
            &[ResolvedFrame::named(format!("native 0x{native_pc:x}"))],
            category,
            false,
        );
        return;
    };
    let mut chain = module
        .resolver
        .as_ref()
        .map(|resolver| resolver.resolve(native_pc))
        .unwrap_or_default();
    if module.generated {
        replace_block_frame(&mut chain, guest_resolver);
    }
    let module_index = module_index.unwrap_or(u32::MAX) as usize + 1;
    emit_resolved_pc(
        profile,
        native_symbols,
        strings,
        output,
        module.library,
        module_index,
        native_pc,
        chain,
        category,
        false,
        "native",
    );
}

#[allow(clippy::too_many_arguments)]
fn emit_resolved_pc(
    profile: &mut Profile,
    native_symbols: &mut HashMap<(usize, u32), fxprof_processed_profile::NativeSymbolHandle>,
    strings: &mut HashMap<String, fxprof_processed_profile::StringHandle>,
    output: &mut Vec<fxprof_processed_profile::FrameHandle>,
    library: fxprof_processed_profile::LibraryHandle,
    library_tag: usize,
    pc: u64,
    mut chain: Vec<ResolvedFrame>,
    category: SubcategoryHandle,
    is_return_address: bool,
    kind: &str,
) {
    let Ok(relative_pc) = u32::try_from(pc) else {
        return;
    };
    if chain.is_empty() {
        chain.push(ResolvedFrame::named(format!("{kind} 0x{pc:x}")));
    }
    emit_frame_chain(
        profile,
        native_symbols,
        strings,
        output,
        library,
        library_tag,
        relative_pc,
        &chain,
        category,
        is_return_address,
    );
}

#[derive(Clone, Debug)]
struct ResolvedFrame {
    name: String,
    file: Option<String>,
    line: Option<u32>,
    column: Option<u32>,
}

impl ResolvedFrame {
    fn named(name: String) -> Self {
        Self {
            name,
            file: None,
            line: None,
            column: None,
        }
    }
}

struct NativeModule {
    resolver: Option<BinaryResolver>,
    library: fxprof_processed_profile::LibraryHandle,
    generated: bool,
}

impl NativeModule {
    fn is_guest_execution_frame(&self, pc: u64) -> bool {
        self.generated
            && self
                .resolver
                .as_ref()
                .is_some_and(|resolver| resolver.is_generated_guest_execution_frame(pc))
    }
}

#[derive(Clone, Debug)]
struct SizedSymbol {
    address: u64,
    size: u64,
    name: String,
}

struct BinaryResolver {
    loader: addr2line::Loader,
    symbols: Vec<SizedSymbol>,
}

impl BinaryResolver {
    fn new(path: &Path) -> Result<Self> {
        let data = fs::read(path)
            .with_context(|| format!("failed to read symbols from {}", path.display()))?;
        let object = object::File::parse(&*data)
            .with_context(|| format!("failed to parse symbols from {}", path.display()))?;
        let mut symbols = object
            .symbols()
            .chain(object.dynamic_symbols())
            .filter(|symbol| symbol.kind() == SymbolKind::Text && symbol.size() != 0)
            .filter_map(|symbol| {
                let name = symbol.name().ok()?;
                (!name.starts_with(".L")).then(|| SizedSymbol {
                    address: symbol.address(),
                    size: symbol.size(),
                    name: name.to_string(),
                })
            })
            .collect::<Vec<_>>();
        symbols.sort_unstable_by_key(|symbol| symbol.address);
        symbols.dedup_by(|left, right| {
            left.address == right.address && left.size == right.size && left.name == right.name
        });
        Ok(Self {
            loader: addr2line::Loader::new(path).map_err(|error| {
                eyre!("failed to load symbols from {}: {error}", path.display())
            })?,
            symbols,
        })
    }

    fn resolve(&self, pc: u64) -> Vec<ResolvedFrame> {
        let fallback_name = self
            .loader
            .find_symbol(pc)
            .filter(|name| !name.starts_with(".L"))
            .map(str::to_string)
            .or_else(|| self.containing_symbol(pc).map(str::to_string));
        let mut chain = Vec::new();
        if let Ok(mut frames) = self.loader.find_frames(pc) {
            while let Ok(Some(frame)) = frames.next() {
                let name = frame
                    .function
                    .as_ref()
                    .and_then(|function| {
                        function
                            .demangle()
                            .ok()
                            .map(|name| name.into_owned())
                            .or_else(|| function.raw_name().ok().map(|name| name.into_owned()))
                    })
                    .or_else(|| fallback_name.clone());
                let Some(name) = name else {
                    continue;
                };
                let (file, line, column) = frame
                    .location
                    .as_ref()
                    .map(|location| {
                        (
                            location.file.map(sanitize_source_path),
                            location.line,
                            location.column,
                        )
                    })
                    .unwrap_or((None, None, None));
                chain.push(ResolvedFrame {
                    name,
                    file,
                    line,
                    column,
                });
            }
            // addr2line yields the interrupted/inlined leaf first.
            chain.reverse();
        }

        let (location_file, location_line, location_column) = self
            .loader
            .find_location(pc)
            .ok()
            .flatten()
            .map(|location| {
                (
                    location.file.map(sanitize_source_path),
                    location.line,
                    location.column,
                )
            })
            .unwrap_or((None, None, None));
        if chain.is_empty() {
            if let Some(name) = fallback_name {
                chain.push(ResolvedFrame {
                    name,
                    file: location_file,
                    line: location_line,
                    column: location_column,
                });
            }
        } else if let Some(leaf) = chain.last_mut() {
            if leaf.file.is_none() {
                leaf.file = location_file;
            }
            if leaf.line.is_none() {
                leaf.line = location_line;
            }
            if leaf.column.is_none() {
                leaf.column = location_column;
            }
        }
        chain
    }

    fn containing_symbol(&self, pc: u64) -> Option<&str> {
        let index = self.symbols.partition_point(|symbol| symbol.address <= pc);
        self.symbols[..index].iter().rev().find_map(|symbol| {
            (pc < symbol.address.saturating_add(symbol.size)).then_some(symbol.name.as_str())
        })
    }

    fn is_generated_guest_execution_frame(&self, pc: u64) -> bool {
        let resolved = self.resolve(pc);
        resolved
            .last()
            .is_some_and(|frame| is_guest_execution_symbol(&frame.name))
            || self
                .loader
                .find_symbol(pc)
                .or_else(|| self.containing_symbol(pc))
                .is_some_and(is_guest_execution_symbol)
    }
}

fn is_guest_execution_symbol(name: &str) -> bool {
    let name = name.rsplit("::").next().unwrap_or(name);
    name.starts_with("block_0x") || name.starts_with("rv_")
}

fn replace_block_frame(chain: &mut [ResolvedFrame], guest_resolver: &BinaryResolver) {
    for frame in chain {
        let Some(hex_pc) = frame.name.strip_prefix("block_0x") else {
            continue;
        };
        let Ok(pc) = u64::from_str_radix(hex_pc, 16) else {
            continue;
        };
        if let Some(guest_frame) = guest_resolver.resolve(pc).last() {
            // The generated C line is the exact native interrupted location.
            // Only substitute its synthetic block name; do not overwrite it
            // with a guest block-entry source location.
            frame.name.clone_from(&guest_frame.name);
        }
    }
}

fn add_library(
    profile: &mut Profile,
    path: &Path,
    name_override: Option<&str>,
    arch: &str,
) -> Result<fxprof_processed_profile::LibraryHandle> {
    let data = fs::read(path)
        .with_context(|| format!("failed to read library metadata from {}", path.display()))?;
    let object = object::File::parse(&*data)
        .with_context(|| format!("failed to parse library metadata from {}", path.display()))?;
    let build_id = object.build_id().ok().flatten();
    let name = name_override
        .and_then(|name| Path::new(name).file_name())
        .or_else(|| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("openvm-profile")
        .to_string();
    let debug_id = build_id
        .map(|build_id| debug_id_from_build_id(build_id, object.is_little_endian()))
        .unwrap_or_default();
    let code_id = build_id.map(|id| debugid::CodeId::from_binary(id).to_string());
    // The payload is already locally symbolicated. Publishing only the stable
    // module name avoids leaking workstation and temporary-directory paths.
    Ok(profile.add_lib(LibraryInfo {
        name: name.clone(),
        debug_name: name.clone(),
        path: name.clone(),
        debug_path: name,
        debug_id,
        code_id,
        arch: Some(arch.to_string()),
    }))
}

fn add_unresolved_library(
    profile: &mut Profile,
    name: &str,
    arch: &str,
) -> fxprof_processed_profile::LibraryHandle {
    let name = Path::new(name)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unavailable-native-module")
        .to_string();
    profile.add_lib(LibraryInfo {
        name: name.clone(),
        debug_name: name.clone(),
        path: name.clone(),
        debug_path: name,
        debug_id: debugid::DebugId::nil(),
        code_id: None,
        arch: Some(arch.to_string()),
    })
}

fn debug_id_from_build_id(build_id: &[u8], little_endian: bool) -> debugid::DebugId {
    debugid::DebugId::from_identifier(build_id, little_endian)
}

fn observed_interval_ns(raw: &RawGuestProfile) -> f64 {
    let duration = raw.end_wall_time_ns.saturating_sub(raw.start_wall_time_ns);
    if duration != 0 && raw.delivered_samples != 0 {
        return duration as f64 / raw.delivered_samples as f64;
    }
    let mut deltas = raw
        .samples
        .windows(2)
        .filter_map(|window| {
            let delta = window[1]
                .wall_time_ns
                .saturating_sub(window[0].wall_time_ns);
            (delta != 0).then_some(delta)
        })
        .collect::<Vec<_>>();
    if !deltas.is_empty() {
        deltas.sort_unstable();
        return deltas[deltas.len() / 2] as f64;
    }
    (1_000_000_000_u64 / u64::from(raw.requested_sample_hz.max(1))) as f64
}

fn sanitize_source_path(path: &str) -> String {
    let components = Path::new(path)
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(str::to_string),
            _ => None,
        })
        .collect::<Vec<_>>();
    if let Some(index) = subsequence(&components, &[".cargo", "registry", "src"]) {
        return components
            .iter()
            .skip(index + 4)
            .cloned()
            .collect::<PathBuf>()
            .display()
            .to_string();
    }
    if let Some(index) = subsequence(&components, &[".cargo", "git", "checkouts"]) {
        let repo = components
            .get(index + 3)
            .map_or("git-checkout", String::as_str);
        let tail = components
            .iter()
            .skip(index + 5)
            .cloned()
            .collect::<PathBuf>();
        return Path::new(repo).join(tail).display().to_string();
    }
    if let Some(index) = subsequence(&components, &["lib", "rustlib", "src", "rust"]) {
        return components
            .iter()
            .skip(index + 3)
            .cloned()
            .collect::<PathBuf>()
            .display()
            .to_string();
    }
    if let Some(index) = components.iter().position(|component| {
        matches!(
            component.as_str(),
            "crates" | "benches" | "guest-programs" | "extensions"
        )
    }) {
        return components[index..]
            .iter()
            .cloned()
            .collect::<PathBuf>()
            .display()
            .to_string();
    }
    if matches!(
        components.first().map(String::as_str),
        Some("home" | "Users")
    ) {
        return join_sanitized_tail(&components, 2, "home-source");
    }
    if components
        .first()
        .is_some_and(|component| component == "tmp")
    {
        return sanitize_temporary_tail(&components, 1);
    }
    if components
        .get(..2)
        .is_some_and(|prefix| prefix[0] == "private" && prefix[1] == "tmp")
    {
        return sanitize_temporary_tail(&components, 2);
    }
    components
        .iter()
        .rev()
        .take(3)
        .rev()
        .cloned()
        .collect::<PathBuf>()
        .display()
        .to_string()
}

fn sanitize_temporary_tail(components: &[String], prefix_len: usize) -> String {
    let remaining = components.len().saturating_sub(prefix_len);
    if remaining >= 2 {
        join_sanitized_tail(components, prefix_len + 1, "temporary-source")
    } else if components
        .last()
        .and_then(|component| Path::new(component).extension())
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| {
            matches!(
                extension,
                "rs" | "c" | "h" | "cc" | "cpp" | "s" | "S" | "asm"
            )
        })
    {
        join_sanitized_tail(components, prefix_len, "temporary-source")
    } else {
        "temporary-source".to_string()
    }
}

fn join_sanitized_tail(components: &[String], start: usize, fallback: &str) -> String {
    let tail = components.iter().skip(start).cloned().collect::<PathBuf>();
    if tail.as_os_str().is_empty() {
        fallback.to_string()
    } else {
        tail.display().to_string()
    }
}

fn subsequence(components: &[String], needle: &[&str]) -> Option<usize> {
    components
        .windows(needle.len())
        .position(|window| window.iter().map(String::as_str).eq(needle.iter().copied()))
}

#[allow(clippy::too_many_arguments)]
fn emit_frame_chain(
    profile: &mut Profile,
    native_symbols: &mut HashMap<(usize, u32), fxprof_processed_profile::NativeSymbolHandle>,
    strings: &mut HashMap<String, fxprof_processed_profile::StringHandle>,
    output: &mut Vec<fxprof_processed_profile::FrameHandle>,
    library: fxprof_processed_profile::LibraryHandle,
    library_tag: usize,
    pc: u32,
    chain: &[ResolvedFrame],
    category: SubcategoryHandle,
    is_return_address: bool,
) {
    if chain.is_empty() {
        return;
    }
    let outer_name = intern_string(profile, strings, &chain[0].name);
    let native_symbol = *native_symbols
        .entry((library_tag, pc))
        .or_insert_with(|| profile.handle_for_native_symbol(library, pc, None, outer_name));
    for (inline_depth, frame) in chain.iter().enumerate() {
        let address = if is_return_address {
            FrameAddress::RelativeAddressFromAdjustedReturnAddress(library, pc)
        } else {
            FrameAddress::RelativeAddressFromInstructionPointer(library, pc)
        };
        let symbol = FrameSymbolInfo {
            name: Some(intern_string(profile, strings, &frame.name)),
            native_symbol,
            source_location: SourceLocation {
                file_path: frame
                    .file
                    .as_deref()
                    .map(|file| intern_string(profile, strings, file)),
                line: frame.line,
                col: frame.column,
                function_start_line: None,
                function_start_col: None,
            },
        };
        output.push(profile.handle_for_frame_with_address_and_symbol(
            address,
            symbol,
            inline_depth as u16,
            category,
            FrameFlags::empty(),
        ));
    }
}

fn intern_string(
    profile: &mut Profile,
    strings: &mut HashMap<String, fxprof_processed_profile::StringHandle>,
    value: &str,
) -> fxprof_processed_profile::StringHandle {
    *strings
        .entry(value.to_string())
        .or_insert_with(|| profile.handle_for_string(value))
}

fn compress_profile(profile: &Profile) -> Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    serde_json::to_writer(&mut encoder, profile).context("failed to serialize Firefox profile")?;
    encoder
        .finish()
        .context("failed to compress Firefox profile")
}

fn upload_profile(compressed_profile: &[u8]) -> Result<String> {
    let compressed_profile = compressed_profile.to_vec();
    std::thread::Builder::new()
        .name("firefox-profile-upload".to_string())
        .spawn(move || upload_profile_blocking(&compressed_profile))
        .context("failed to start Firefox Profiler upload thread")?
        .join()
        .map_err(|_| eyre!("Firefox Profiler upload thread panicked"))?
}

fn upload_profile_blocking(compressed_profile: &[u8]) -> Result<String> {
    upload_profile_blocking_to(DEFAULT_UPLOAD_URL, compressed_profile)
}

fn upload_profile_blocking_to(upload_url: &str, compressed_profile: &[u8]) -> Result<String> {
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .context("failed to create Firefox Profiler upload client")?;
    let mut last_error = None;
    for (attempt, backoff) in [
        Duration::from_secs(5),
        Duration::from_secs(15),
        Duration::ZERO,
    ]
    .into_iter()
    .enumerate()
    {
        let result = client
            .post(upload_url)
            .header(ACCEPT, FIREFOX_ACCEPT)
            .body(compressed_profile.to_vec())
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::text);
        match result {
            Ok(response) => return public_url_from_response(response.trim()),
            Err(error) => {
                last_error = Some(error);
                if !backoff.is_zero() {
                    eprintln!(
                        "[openvm] Firefox Profiler upload failed (attempt {}/3); retrying in {}s",
                        attempt + 1,
                        backoff.as_secs()
                    );
                    std::thread::sleep(backoff);
                }
            }
        }
    }
    Err(last_error
        .map(|error| eyre!(error))
        .unwrap_or_else(|| eyre!("Firefox Profiler upload failed")))
    .context("failed to upload execution profile after 3 attempts")
}

fn public_url_from_response(response: &str) -> Result<String> {
    let parts = response.split('.').collect::<Vec<_>>();
    if parts.len() != 3 {
        bail!("unexpected Firefox Profiler response");
    }
    let payload = parts[1];
    let decoded = URL_SAFE_NO_PAD
        .decode(payload)
        .context("invalid Firefox Profiler response payload")?;
    let decoded: Value =
        serde_json::from_slice(&decoded).context("invalid Firefox Profiler response JSON")?;
    let token = decoded
        .get("profileToken")
        .and_then(Value::as_str)
        .filter(|token| !token.is_empty())
        .ok_or_else(|| eyre!("Firefox Profiler response did not contain profileToken"))?;
    Ok(format!("https://profiler.firefox.com/public/{token}"))
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{Read, Write},
        net::TcpListener,
        thread,
    };

    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

    use super::{
        build_firefox_profile, debug_id_from_build_id, is_guest_execution_symbol,
        observed_interval_ns, parse_raw_profile, public_url_from_response, sanitize_source_path,
        upload_profile_blocking_to, FIREFOX_ACCEPT,
    };

    #[test]
    fn parses_v3_capture_metadata_and_explicit_stack_roles() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.raw");
        fs::write(
            &path,
            r#"{
                "version":3,
                "requested_sample_hz":2000,
                "owner_tid":41,
                "start_unix_time_ns":1000000000,
                "start_wall_time_ns":100,
                "end_wall_time_ns":1100,
                "start_cpu_time_ns":50,
                "end_cpu_time_ns":950,
                "delivered_samples":1,
                "dropped_samples":0,
                "timer_overruns":0,
                "timer_arm_failures":0,
                "clock_failures":0,
                "native_modules":[{"name":"libguest.so","path":"/tmp/run-123/libguest.so","generated":true}],
                "samples":[{
                    "wall_time_ns":400,
                    "cpu_time_ns":300,
                    "native_leaf":{"module_index":0,"pc":4660},
                    "guest_callsite_pc":64,
                    "guest_return_pcs":[16,32],
                    "stack_truncated":false
                }]
            }"#,
        )
        .unwrap();
        let raw = parse_raw_profile(&path).unwrap();
        assert_eq!(raw.owner_tid, 41);
        assert_eq!(raw.native_modules[0].name, "libguest.so");
        assert_eq!(raw.samples[0].guest_return_pcs, [16, 32]);
        assert_eq!(raw.samples[0].guest_callsite_pc, Some(64));
        assert_eq!(raw.samples[0].native_leaf.as_ref().unwrap().pc, 0x1234);
    }

    #[test]
    fn interval_comes_from_observed_capture_duration() {
        let raw = openvm_circuit::arch::rvr::RawGuestProfile {
            version: 3,
            requested_sample_hz: 10_000,
            owner_tid: 1,
            start_unix_time_ns: 0,
            start_wall_time_ns: 1_000,
            end_wall_time_ns: 5_001_000,
            start_cpu_time_ns: 0,
            end_cpu_time_ns: 0,
            delivered_samples: 5,
            dropped_samples: 0,
            timer_overruns: 0,
            timer_arm_failures: 0,
            clock_failures: 0,
            native_modules: vec![],
            samples: vec![],
        };
        assert_eq!(observed_interval_ns(&raw), 1_000_000.0);
    }

    #[test]
    fn firefox_metadata_uses_capture_clock_and_hides_binary_path() {
        let executable = std::env::current_exe().unwrap();
        let raw: openvm_circuit::arch::rvr::RawGuestProfile = serde_json::from_str(
            r#"{
                "version":3,
                "requested_sample_hz":10000,
                "owner_tid":7,
                "start_unix_time_ns":1234000000,
                "start_wall_time_ns":1000000,
                "end_wall_time_ns":3000000,
                "start_cpu_time_ns":500000,
                "end_cpu_time_ns":2500000,
                "delivered_samples":2,
                "dropped_samples":0,
                "timer_overruns":0,
                "timer_arm_failures":0,
                "clock_failures":0,
                "native_modules":[],
                "samples":[
                    {"wall_time_ns":1500000,"cpu_time_ns":1000000,"native_leaf":null,"guest_callsite_pc":null,"guest_return_pcs":[1],"stack_truncated":false},
                    {"wall_time_ns":2500000,"cpu_time_ns":2000000,"native_leaf":null,"guest_callsite_pc":null,"guest_return_pcs":[1],"stack_truncated":false}
                ]
            }"#,
        )
        .unwrap();
        let profile = build_firefox_profile(&raw, &executable, None).unwrap();
        let json = serde_json::to_value(profile).unwrap();
        assert_eq!(json["meta"]["interval"], 1.0);
        assert_eq!(json["meta"]["startTime"], 1234.0);
        let library = &json["libs"][0];
        assert_eq!(library["path"], library["name"]);
        assert!(!library["path"].as_str().unwrap().contains('/'));
        assert_eq!(library["arch"], "riscv64");
    }

    #[test]
    fn sanitizes_private_and_dependency_source_prefixes() {
        assert_eq!(
            sanitize_source_path(
                "/home/alice/.cargo/registry/src/index-abcd/serde-1.0.0/src/lib.rs"
            ),
            "serde-1.0.0/src/lib.rs"
        );
        assert_eq!(
            sanitize_source_path("/tmp/build.secret/openvm/crates/vm/src/lib.rs"),
            "crates/vm/src/lib.rs"
        );
        assert_eq!(
            sanitize_source_path("/home/alice/private/generated/block_0x10.c"),
            "private/generated/block_0x10.c"
        );
        assert_eq!(sanitize_source_path("/home/alice/main.rs"), "main.rs");
        assert_eq!(
            sanitize_source_path("/Users/alice/project/src/main.rs"),
            "project/src/main.rs"
        );
        assert_eq!(
            sanitize_source_path("/tmp/openvm-run.ABC123/dispatch.c"),
            "dispatch.c"
        );
        assert_eq!(
            sanitize_source_path("/private/tmp/.tmp-secret/block.c"),
            "block.c"
        );
        assert_eq!(
            sanitize_source_path("/tmp/openvm-run.ABC123"),
            "temporary-source"
        );
    }

    #[test]
    fn build_id_debug_id_uses_object_endianness() {
        let identifier = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10,
        ];
        assert_eq!(
            debug_id_from_build_id(&identifier, true)
                .breakpad()
                .to_string(),
            "0403020106050807090A0B0C0D0E0F100"
        );
        assert_eq!(
            debug_id_from_build_id(&identifier, false)
                .breakpad()
                .to_string(),
            "0102030405060708090A0B0C0D0E0F100"
        );
    }

    #[test]
    fn distinguishes_generated_guest_frames_from_host_helpers() {
        assert!(is_guest_execution_symbol("block_0x00200100"));
        assert!(is_guest_execution_symbol("block_0x00200100_checkpoint"));
        assert!(is_guest_execution_symbol("rv_execute"));
        assert!(!is_guest_execution_symbol("metered_checkpoint"));
        assert!(!is_guest_execution_symbol("openvm_hint_input"));
        assert!(!is_guest_execution_symbol("memcpy"));
    }

    #[test]
    fn extracts_public_url_from_upload_jwt() {
        let payload = URL_SAFE_NO_PAD.encode(r#"{"profileToken":"test-token"}"#);
        let response = format!("header.{payload}.signature");
        assert_eq!(
            public_url_from_response(&response).unwrap(),
            "https://profiler.firefox.com/public/test-token"
        );
    }

    #[test]
    fn rejects_malformed_upload_responses() {
        assert!(public_url_from_response("not-a-jwt").is_err());
        assert!(public_url_from_response("header.payload.signature.extra").is_err());

        let payload = URL_SAFE_NO_PAD.encode(r#"{"profileToken":""}"#);
        let response = format!("header.{payload}.signature");
        assert!(public_url_from_response(&response).is_err());
    }

    #[test]
    fn posts_the_compressed_profile_using_the_firefox_protocol() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let upload_url = format!("http://{}/compressed-store", listener.local_addr().unwrap());
        let compressed_profile = b"\x1f\x8bopenvm-profile";
        let server = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0; 1024];
            let (header_end, content_length) = loop {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "request ended before its headers");
                request.extend_from_slice(&buffer[..count]);
                let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n")
                else {
                    continue;
                };
                let headers = String::from_utf8_lossy(&request[..header_end]);
                let content_length = headers
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(name, value)| {
                            if name.eq_ignore_ascii_case("content-length") {
                                Some(value.trim().parse::<usize>().unwrap())
                            } else {
                                None
                            }
                        })
                    })
                    .expect("content-length header");
                break (header_end + 4, content_length);
            };
            while request.len() < header_end + content_length {
                let count = stream.read(&mut buffer).unwrap();
                assert_ne!(count, 0, "request ended before its body");
                request.extend_from_slice(&buffer[..count]);
            }

            let payload = URL_SAFE_NO_PAD.encode(r#"{"profileToken":"protocol-test"}"#);
            let jwt = format!("header.{payload}.signature");
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{jwt}",
                jwt.len()
            )
            .unwrap();
            request
        });

        let public_url = upload_profile_blocking_to(&upload_url, compressed_profile).unwrap();
        assert_eq!(
            public_url,
            "https://profiler.firefox.com/public/protocol-test"
        );

        let request = server.join().unwrap();
        let header_end = request
            .windows(4)
            .position(|bytes| bytes == b"\r\n\r\n")
            .unwrap();
        let headers = String::from_utf8_lossy(&request[..header_end]);
        assert!(headers.starts_with("POST /compressed-store HTTP/1.1\r\n"));
        assert!(headers.lines().any(|line| {
            line.split_once(':').is_some_and(|(name, value)| {
                name.eq_ignore_ascii_case("accept") && value.trim() == FIREFOX_ACCEPT
            })
        }));
        assert_eq!(&request[header_end + 4..], compressed_profile);
    }
}
