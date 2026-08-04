use std::{
    fs,
    path::{Path, PathBuf},
    thread::{self, Builder},
    time::{Duration, SystemTime},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use eyre::{bail, eyre, Context, Result};
use flate2::{write::GzEncoder, Compression};
use fxprof_processed_profile::{
    CategoryColor, CategoryPairHandle, CpuDelta, Frame, FrameFlags, FrameInfo, Profile,
    SamplingInterval, Timestamp,
};
use object::{Object, ObjectSymbol, SymbolKind};
use openvm_circuit::arch::rvr::{GuestProfileConfig, RawGuestProfile, RAW_GUEST_PROFILE_VERSION};
use reqwest::{blocking::Client, header::ACCEPT};
use serde_json::Value;

const DEFAULT_UPLOAD_URL: &str = "https://api.profiler.firefox.com/compressed-store";
const FIREFOX_ACCEPT: &str = "application/vnd.firefox-profiler+json;version=1.0";

/// Coordinates one guest execution capture and its Firefox profile conversion.
pub struct FirefoxProfiler {
    config: GuestProfileConfig,
    _raw_profile: tempfile::NamedTempFile,
    guest_elf: tempfile::NamedTempFile,
    native_artifact: tempfile::NamedTempFile,
}

impl FirefoxProfiler {
    pub fn new(guest_elf_path: impl Into<PathBuf>, sample_hz: u32) -> Result<Self> {
        let guest_elf_path = guest_elf_path.into();
        let raw_profile =
            tempfile::NamedTempFile::new().context("failed to create temporary RVR profile")?;
        let guest_elf = tempfile::NamedTempFile::new()
            .context("failed to create temporary guest profile artifact")?;
        fs::copy(&guest_elf_path, guest_elf.path()).with_context(|| {
            format!(
                "failed to preserve guest profile artifact {}",
                guest_elf_path.display()
            )
        })?;
        let native_artifact = tempfile::NamedTempFile::new()
            .context("failed to create temporary native profile artifact")?;
        let config = GuestProfileConfig::raw_session_with_native_artifact(
            raw_profile.path(),
            native_artifact.path(),
            sample_hz,
        )
        .map_err(|error| eyre!(error))?;
        Ok(Self {
            config,
            _raw_profile: raw_profile,
            guest_elf,
            native_artifact,
        })
    }

    /// Explicit VM/SDK configuration for the execution being profiled.
    pub fn config(&self) -> &GuestProfileConfig {
        &self.config
    }

    /// Convert the captured raw samples into a reusable Firefox artifact.
    pub fn finish(self) -> Result<FirefoxProfile> {
        self.config.finish_session().map_err(|error| eyre!(error))?;
        FirefoxProfile::from_raw_guest_stacks(
            self.config.output(),
            self.guest_elf.path(),
            Some(self.native_artifact.path()),
        )
    }
}

/// A generated, symbolicated Firefox Profiler artifact.
pub struct FirefoxProfile {
    compressed: Vec<u8>,
    sample_count: usize,
}

impl FirefoxProfile {
    /// Convert an ordered v3 raw guest profile into a function-level Firefox profile.
    ///
    /// Raw v3 retains exact PCs and return-address roles. The processed profile
    /// intentionally aggregates frames by function name for stable call trees.
    ///
    /// If supplied, `native_artifact_path` overrides the generated module path
    /// recorded in the raw profile.
    pub fn from_raw_guest_stacks(
        raw_profile_path: &Path,
        guest_elf_path: &Path,
        native_artifact_path: Option<&Path>,
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

    let mut native_modules = Vec::with_capacity(raw.native_modules.len());
    for module in &raw.native_modules {
        let path = if module.generated {
            native_artifact_path.unwrap_or_else(|| Path::new(&module.path))
        } else {
            Path::new(&module.path)
        };
        match BinaryResolver::new(path) {
            Ok(resolver) => {
                native_modules.push(NativeModule {
                    resolver: Some(resolver),
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
                    generated: false,
                });
            }
        }
    }
    let guest_category: CategoryPairHandle =
        profile.add_category("Guest", CategoryColor::Yellow).into();
    let mut previous_cpu_time = raw.start_cpu_time_ns;

    for sample in &raw.samples {
        let timestamp_ns = sample.wall_time_ns.saturating_sub(raw.start_wall_time_ns);
        let timestamp = Timestamp::from_millis_since_reference(timestamp_ns as f64 / 1_000_000.0);
        let cpu_delta = CpuDelta::from_nanos(sample.cpu_time_ns.saturating_sub(previous_cpu_time));
        previous_cpu_time = sample.cpu_time_ns;
        let mut frame_handles = Vec::new();

        // The outermost frame's saved return address is zero by ABI convention.
        // It is a stack terminator, not guest address zero; resolving it would
        // incorrectly attribute every sample to whichever symbol starts at 0.
        for &return_pc in sample.guest_return_pcs.iter().filter(|&&pc| pc != 0) {
            let lookup_pc = return_pc.saturating_sub(1);
            emit_resolved_pc(
                &mut profile,
                &mut frame_handles,
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
                    &mut frame_handles,
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
                &mut frame_handles,
                &native_modules,
                &guest_resolver,
                native_leaf.module_index,
                native_leaf.pc,
                guest_category,
            );
        }

        let stack = profile.intern_stack_frames(thread, frame_handles.into_iter());
        profile.add_sample(thread, timestamp, stack, cpu_delta, 1);
    }

    let end_ns = raw.end_wall_time_ns.saturating_sub(raw.start_wall_time_ns);
    let end = Timestamp::from_millis_since_reference(end_ns as f64 / 1_000_000.0);
    profile.set_process_end_time(process, end);
    profile.set_thread_end_time(thread, end);
    Ok(profile)
}

fn emit_native_leaf(
    profile: &mut Profile,
    output: &mut Vec<FrameInfo>,
    modules: &[NativeModule],
    guest_resolver: &BinaryResolver,
    module_index: Option<u32>,
    native_pc: u64,
    category: CategoryPairHandle,
) {
    let Some(module) = module_index.and_then(|index| modules.get(index as usize)) else {
        let pc = u32::try_from(native_pc).unwrap_or(u32::MAX);
        emit_frame_chain(
            profile,
            output,
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
    emit_resolved_pc(profile, output, native_pc, chain, category, false, "native");
}

fn emit_resolved_pc(
    profile: &mut Profile,
    output: &mut Vec<FrameInfo>,
    pc: u64,
    mut chain: Vec<ResolvedFrame>,
    category: CategoryPairHandle,
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
        output,
        relative_pc,
        &chain,
        category,
        is_return_address,
    );
}

#[derive(Clone, Debug)]
struct ResolvedFrame {
    name: String,
}

impl ResolvedFrame {
    fn named(name: String) -> Self {
        Self { name }
    }
}

struct NativeModule {
    resolver: Option<BinaryResolver>,
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
        let debug_path = object
            .build_id()
            .ok()
            .flatten()
            .and_then(build_id_debug_path)
            .filter(|path| path.is_file());
        let resolver_path = debug_path.as_deref().unwrap_or(path);
        let resolver_data = if debug_path.is_some() {
            fs::read(resolver_path).with_context(|| {
                format!(
                    "failed to read debug symbols from {}",
                    resolver_path.display()
                )
            })?
        } else {
            data
        };
        let resolver_object = object::File::parse(&*resolver_data).with_context(|| {
            format!(
                "failed to parse debug symbols from {}",
                resolver_path.display()
            )
        })?;
        let mut symbols = resolver_object
            .symbols()
            .chain(resolver_object.dynamic_symbols())
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
            loader: addr2line::Loader::new(resolver_path).map_err(|error| {
                eyre!(
                    "failed to load symbols from {}: {error}",
                    resolver_path.display()
                )
            })?,
            symbols,
        })
    }

    fn resolve(&self, pc: u64) -> Vec<ResolvedFrame> {
        // `addr2line::Loader::find_symbol` returns the nearest preceding
        // symbol, even when that symbol does not contain `pc`. On stripped
        // DSOs this can assign an unrelated public name to an internal
        // function. Only use a sized symbol whose range contains the PC.
        let fallback_name = self.containing_symbol(pc).map(str::to_string);
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
                chain.push(ResolvedFrame { name });
            }
            // addr2line yields the interrupted/inlined leaf first.
            chain.reverse();
        }

        if chain.is_empty() {
            if let Some(name) = fallback_name {
                chain.push(ResolvedFrame { name });
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
                .containing_symbol(pc)
                .is_some_and(is_guest_execution_symbol)
    }
}

fn build_id_debug_path(build_id: &[u8]) -> Option<PathBuf> {
    let (&first, rest) = build_id.split_first()?;
    let file = rest
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Some(
        Path::new("/usr/lib/debug/.build-id")
            .join(format!("{first:02x}"))
            .join(format!("{file}.debug")),
    )
}

fn is_guest_execution_symbol(name: &str) -> bool {
    let name = name.rsplit("::").next().unwrap_or(name);
    name.starts_with("block_0x") || name.starts_with("rv_")
}

fn replace_block_frame(chain: &mut [ResolvedFrame], guest_resolver: &BinaryResolver) {
    for frame in chain {
        let Some(pc) = block_symbol_pc(&frame.name) else {
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

fn block_symbol_pc(name: &str) -> Option<u64> {
    let suffix = name.strip_prefix("block_0x")?;
    let hex_len = suffix
        .as_bytes()
        .iter()
        .take_while(|byte| byte.is_ascii_hexdigit())
        .count();
    u64::from_str_radix(suffix.get(..hex_len)?, 16).ok()
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

fn emit_frame_chain(
    profile: &mut Profile,
    output: &mut Vec<FrameInfo>,
    _pc: u32,
    chain: &[ResolvedFrame],
    category: CategoryPairHandle,
    _is_return_address: bool,
) {
    for frame in chain {
        output.push(FrameInfo {
            frame: Frame::Label(profile.intern_string(&frame.name)),
            category_pair: category,
            flags: FrameFlags::empty(),
        });
    }
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
    Builder::new()
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
                    thread::sleep(backoff);
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
        env, fs,
        io::{Read, Write},
        net::TcpListener,
        path::PathBuf,
        thread,
    };

    use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};

    use super::{
        block_symbol_pc, build_firefox_profile, build_id_debug_path, is_guest_execution_symbol,
        observed_interval_ns, parse_raw_profile, public_url_from_response,
        upload_profile_blocking_to, RawGuestProfile, FIREFOX_ACCEPT,
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
        let raw = RawGuestProfile {
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
        let executable = env::current_exe().unwrap();
        let raw: RawGuestProfile = serde_json::from_str(
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
        assert!(!json.to_string().contains(&executable.display().to_string()));
    }

    #[test]
    fn zero_return_pc_is_treated_as_a_stack_terminator() {
        let executable = env::current_exe().unwrap();
        let raw: RawGuestProfile = serde_json::from_str(
            r#"{
                "version":3,
                "requested_sample_hz":1000,
                "owner_tid":7,
                "start_unix_time_ns":1000000000,
                "start_wall_time_ns":1000000,
                "end_wall_time_ns":2000000,
                "start_cpu_time_ns":500000,
                "end_cpu_time_ns":1500000,
                "delivered_samples":1,
                "dropped_samples":0,
                "timer_overruns":0,
                "timer_arm_failures":0,
                "clock_failures":0,
                "native_modules":[],
                "samples":[
                    {"wall_time_ns":1500000,"cpu_time_ns":1000000,"native_leaf":null,"guest_callsite_pc":null,"guest_return_pcs":[0],"stack_truncated":false}
                ]
            }"#,
        )
        .unwrap();
        let profile = build_firefox_profile(&raw, &executable, None).unwrap();
        let json = serde_json::to_value(profile).unwrap();
        assert!(json["threads"][0]["samples"]["stack"][0].is_null());
    }

    #[test]
    fn build_id_maps_to_the_standard_debug_file_path() {
        assert_eq!(
            build_id_debug_path(&[0x8e, 0x9f, 0xd8, 0x27]).unwrap(),
            PathBuf::from("/usr/lib/debug/.build-id/8e/9fd827.debug")
        );
        assert_eq!(build_id_debug_path(&[]), None);
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
    fn parses_checkpoint_block_symbols() {
        assert_eq!(block_symbol_pc("block_0x00200100"), Some(0x0020_0100));
        assert_eq!(
            block_symbol_pc("block_0x00200100_checkpoint"),
            Some(0x0020_0100)
        );
        assert_eq!(block_symbol_pc("block_0x"), None);
        assert_eq!(block_symbol_pc("rv_execute"), None);
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
