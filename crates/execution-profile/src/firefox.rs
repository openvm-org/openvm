use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
    time::{Duration, SystemTime},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use eyre::{bail, eyre, Context, Result};
use flate2::{write::GzEncoder, Compression};
use fxprof_processed_profile::{
    Category, CategoryColor, CpuDelta, FrameAddress, FrameFlags, FrameSymbolInfo, LibraryInfo,
    Profile, SamplingInterval, SourceLocation, SubcategoryHandle, Timestamp,
};
use openvm_circuit::arch::rvr::{
    default_addr2line_cmd, GuestDebugMap, GuestProfileConfig, RawGuestProfile,
    RawGuestProfileSample, RAW_GUEST_PROFILE_VERSION,
};
use reqwest::{blocking::Client, header::ACCEPT};
use serde_json::Value;

const DEFAULT_UPLOAD_URL: &str = "https://api.profiler.firefox.com/compressed-store";
const FIREFOX_ACCEPT: &str = "application/vnd.firefox-profiler+json;version=1.0";

/// Coordinates one guest execution capture and its Firefox profile conversion.
pub struct FirefoxProfiler {
    guest_elf_path: PathBuf,
    config: GuestProfileConfig,
    _raw_profile: tempfile::NamedTempFile,
    native_artifact: tempfile::NamedTempFile,
}

impl FirefoxProfiler {
    pub fn new(guest_elf_path: impl Into<PathBuf>, sample_hz: u32) -> Result<Self> {
        let raw_profile =
            tempfile::NamedTempFile::new().context("failed to create temporary RVR profile")?;
        let native_artifact = tempfile::NamedTempFile::new()
            .context("failed to create temporary native RVR artifact")?;
        let config = GuestProfileConfig::raw_with_native_artifact(
            raw_profile.path(),
            native_artifact.path(),
            sample_hz,
        )
        .map_err(|error| eyre!(error))?;
        Ok(Self {
            guest_elf_path: guest_elf_path.into(),
            config,
            _raw_profile: raw_profile,
            native_artifact,
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
            Some(self.native_artifact.path()),
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
    /// Convert ordered raw guest stacks into a symbolicated Firefox profile.
    pub fn from_raw_guest_stacks(
        raw_profile_path: &Path,
        guest_elf_path: &Path,
        native_artifact_path: Option<&Path>,
        sample_hz: u32,
    ) -> Result<Self> {
        let samples = parse_raw_samples(raw_profile_path)?;
        let profile =
            build_firefox_profile(&samples, guest_elf_path, native_artifact_path, sample_hz)?;
        Ok(Self {
            compressed: compress_profile(&profile)?,
            sample_count: samples.len(),
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

fn parse_raw_samples(path: &Path) -> Result<Vec<RawGuestProfileSample>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read RVR samples from {}", path.display()))?;
    let trimmed = contents.trim();
    let samples = if trimmed.starts_with('{') {
        let raw: RawGuestProfile =
            serde_json::from_str(trimmed).context("failed to parse versioned RVR profile")?;
        if raw.version != RAW_GUEST_PROFILE_VERSION {
            bail!(
                "unsupported RVR profile version {}; expected {}",
                raw.version,
                RAW_GUEST_PROFILE_VERSION
            );
        }
        raw.samples
    } else {
        // Preserve support for profiles captured before host IPs and clocks
        // were added. Their attribution and timeline are necessarily less
        // precise, but existing artifacts remain convertible.
        let mut samples = Vec::new();
        for (line_index, line) in contents.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let guest_pcs = line
                .split(';')
                .map(|value| {
                    let value = value.trim().strip_prefix("0x").unwrap_or(value.trim());
                    u64::from_str_radix(value, 16).with_context(|| {
                        format!("invalid guest PC on profile line {}", line_index + 1)
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            if !guest_pcs.is_empty() {
                samples.push(RawGuestProfileSample {
                    wall_time_ns: 0,
                    cpu_time_ns: 0,
                    host_pc: None,
                    guest_pcs,
                });
            }
        }
        samples
    };
    if samples.is_empty() {
        bail!(
            "RVR execution completed before any profile samples were captured; try a larger workload or a higher --rate"
        );
    }
    Ok(samples)
}

fn build_firefox_profile(
    samples: &[RawGuestProfileSample],
    guest_elf_path: &Path,
    native_artifact_path: Option<&Path>,
    sample_hz: u32,
) -> Result<Profile> {
    let lookup_pcs = samples
        .iter()
        .flat_map(|sample| {
            sample
                .guest_pcs
                .iter()
                .enumerate()
                .filter_map(|(index, &pc)| {
                    let is_return_address =
                        sample.host_pc.is_some() || index + 1 != sample.guest_pcs.len();
                    let pc = if is_return_address {
                        pc.saturating_sub(1)
                    } else {
                        pc
                    };
                    u32::try_from(pc).ok()
                })
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let debug_map = GuestDebugMap::from_elf(guest_elf_path, &lookup_pcs, &default_addr2line_cmd())
        .map_err(|error| eyre!(error))?;

    let interval = SamplingInterval::from_hz(sample_hz as f32);
    let mut profile = Profile::new("OpenVM guest execution", SystemTime::now().into(), interval);
    profile.set_os_name("OpenVM RV64 guest");
    profile.set_symbolicated(true);
    let zero = Timestamp::from_millis_since_reference(0.0);
    let process = profile.add_process("OpenVM guest", 1, zero);
    let thread = profile.add_thread(process, 1, zero, true);
    profile.set_thread_name(thread, "RV64 guest execution");
    profile.add_initial_selected_thread(thread);

    let guest_lib = add_library(&mut profile, guest_elf_path, None, Some("riscv64"));
    let host_resolver = native_artifact_path
        .filter(|path| path.metadata().is_ok_and(|metadata| metadata.len() != 0))
        .map(HostResolver::new)
        .transpose()?;
    let host_lib = native_artifact_path
        .map(|path| add_library(&mut profile, path, Some("libopenvm.so"), None));
    let guest_category: SubcategoryHandle = profile
        .handle_for_category(Category("Guest", CategoryColor::Yellow))
        .into();
    let mut native_symbols = HashMap::new();
    let mut strings = HashMap::new();

    let interval_ms = 1_000.0 / f64::from(sample_hz);
    let first_wall_time = samples
        .iter()
        .find_map(|sample| (sample.wall_time_ns != 0).then_some(sample.wall_time_ns));
    let mut previous_cpu_time = None;
    let mut end_ms = 0.0;
    for (sample_index, sample) in samples.iter().enumerate() {
        let timestamp_ms = first_wall_time
            .filter(|_| sample.wall_time_ns != 0)
            .map_or(sample_index as f64 * interval_ms, |first| {
                sample.wall_time_ns.saturating_sub(first) as f64 / 1_000_000.0
            });
        end_ms = timestamp_ms;
        let cpu_delta = previous_cpu_time
            .filter(|_| sample.cpu_time_ns != 0)
            .map_or(CpuDelta::ZERO, |previous| {
                CpuDelta::from_nanos(sample.cpu_time_ns.saturating_sub(previous))
            });
        if sample.cpu_time_ns != 0 {
            previous_cpu_time = Some(sample.cpu_time_ns);
        }

        let mut frame_handles = Vec::new();
        for (frame_index, &pc) in sample.guest_pcs.iter().enumerate() {
            let is_return_address =
                sample.host_pc.is_some() || frame_index + 1 != sample.guest_pcs.len();
            let lookup_pc = if is_return_address {
                pc.saturating_sub(1)
            } else {
                pc
            };
            let Ok(lookup_pc) = u32::try_from(lookup_pc) else {
                continue;
            };
            let location = debug_map.get(lookup_pc);
            let frame = ResolvedFrame {
                name: location
                    .filter(|location| !location.function.is_empty())
                    .map(|location| location.function.clone())
                    .unwrap_or_else(|| format!("0x{pc:08x}")),
                file: location
                    .filter(|location| !location.file.is_empty() && location.file != "??")
                    .map(|location| location.file.clone()),
                line: location
                    .filter(|location| !location.file.is_empty() && location.line != 0)
                    .map(|location| location.line),
            };
            emit_frame_chain(
                &mut profile,
                &mut native_symbols,
                &mut strings,
                &mut frame_handles,
                guest_lib,
                0,
                lookup_pc,
                std::slice::from_ref(&frame),
                guest_category,
                is_return_address,
            );
        }

        if let Some(host_pc) = sample.host_pc {
            let mut chain = host_resolver
                .as_ref()
                .map(|resolver| resolver.resolve(host_pc))
                .unwrap_or_default();
            replace_block_frame(&mut chain, &debug_map);
            if chain.is_empty() {
                chain.push(ResolvedFrame {
                    name: format!("native 0x{host_pc:x}"),
                    file: None,
                    line: None,
                });
            }
            if let Some(host_lib) = host_lib {
                let host_pc = u32::try_from(host_pc).unwrap_or(u32::MAX);
                emit_frame_chain(
                    &mut profile,
                    &mut native_symbols,
                    &mut strings,
                    &mut frame_handles,
                    host_lib,
                    1,
                    host_pc,
                    &chain,
                    guest_category,
                    false,
                );
            }
        }

        let mut stack = None;
        for frame in frame_handles {
            stack = Some(profile.handle_for_stack(frame, stack));
        }
        profile.add_sample(
            thread,
            Timestamp::from_millis_since_reference(timestamp_ms),
            stack,
            cpu_delta,
            1,
        );
    }
    let end = Timestamp::from_millis_since_reference(end_ms);
    profile.set_process_end_time(process, end);
    profile.set_thread_end_time(thread, end);
    Ok(profile)
}

#[derive(Clone, Debug)]
struct ResolvedFrame {
    name: String,
    file: Option<String>,
    line: Option<u32>,
}

struct HostResolver {
    loader: addr2line::Loader,
}

impl HostResolver {
    fn new(path: &Path) -> Result<Self> {
        Ok(Self {
            loader: addr2line::Loader::new(path).map_err(|error| {
                eyre!(
                    "failed to load native symbols from {}: {error}",
                    path.display()
                )
            })?,
        })
    }

    fn resolve(&self, pc: u64) -> Vec<ResolvedFrame> {
        let mut chain = Vec::new();
        if let Ok(mut frames) = self.loader.find_frames(pc) {
            while let Ok(Some(frame)) = frames.next() {
                let Some(function) = frame.function.as_ref() else {
                    continue;
                };
                let Some(name) = function
                    .demangle()
                    .ok()
                    .map(|name| name.into_owned())
                    .or_else(|| function.raw_name().ok().map(|name| name.into_owned()))
                else {
                    continue;
                };
                let (file, line) = frame
                    .location
                    .as_ref()
                    .map(|location| (location.file.map(str::to_string), location.line))
                    .unwrap_or((None, None));
                chain.push(ResolvedFrame { name, file, line });
            }
            // addr2line yields the interrupted/inlined leaf first.
            chain.reverse();
        }
        if chain.is_empty() {
            if let Some(name) = self.loader.find_symbol(pc) {
                chain.push(ResolvedFrame {
                    name: name.to_string(),
                    file: None,
                    line: None,
                });
            }
        }
        chain
    }
}

fn replace_block_frame(chain: &mut [ResolvedFrame], debug_map: &GuestDebugMap) {
    let Some(frame) = chain.first_mut() else {
        return;
    };
    let Some(hex_pc) = frame.name.strip_prefix("block_0x") else {
        return;
    };
    let Ok(pc) = u32::from_str_radix(hex_pc, 16) else {
        return;
    };
    let Some(location) = debug_map.get(pc) else {
        return;
    };
    if !location.function.is_empty() {
        frame.name.clone_from(&location.function);
    }
    if !location.file.is_empty() && location.file != "??" {
        frame.file = Some(location.file.clone());
        frame.line = (location.line != 0).then_some(location.line);
    }
}

fn add_library(
    profile: &mut Profile,
    path: &Path,
    name_override: Option<&str>,
    arch: Option<&str>,
) -> fxprof_processed_profile::LibraryHandle {
    let name = name_override.map(str::to_string).unwrap_or_else(|| {
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("openvm-profile")
            .to_string()
    });
    profile.add_lib(LibraryInfo {
        name: name.clone(),
        debug_name: name,
        path: path.display().to_string(),
        debug_path: path.display().to_string(),
        debug_id: debugid::DebugId::nil(),
        code_id: None,
        arch: arch.map(str::to_string),
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_frame_chain(
    profile: &mut Profile,
    native_symbols: &mut HashMap<(u8, u32), fxprof_processed_profile::NativeSymbolHandle>,
    strings: &mut HashMap<String, fxprof_processed_profile::StringHandle>,
    output: &mut Vec<fxprof_processed_profile::FrameHandle>,
    library: fxprof_processed_profile::LibraryHandle,
    library_tag: u8,
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
                col: None,
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
        parse_raw_samples, public_url_from_response, upload_profile_blocking_to, FIREFOX_ACCEPT,
    };

    #[test]
    fn parses_legacy_ordered_raw_stacks_without_inventing_native_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.raw");
        fs::write(&path, "0x30;0x20;0x10\n0x40\n").unwrap();
        let samples = parse_raw_samples(&path).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].guest_pcs, vec![0x30, 0x20, 0x10]);
        assert_eq!(samples[1].guest_pcs, vec![0x40]);
        assert!(samples.iter().all(|sample| sample.host_pc.is_none()));
        assert!(samples.iter().all(|sample| sample.wall_time_ns == 0));
    }

    #[test]
    fn parses_versioned_samples_with_native_pc_and_clocks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.raw");
        fs::write(
            &path,
            r#"{"version":2,"samples":[{"wall_time_ns":12000,"cpu_time_ns":9000,"host_pc":4660,"guest_pcs":[48,32]}]}"#,
        )
        .unwrap();
        let samples = parse_raw_samples(&path).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].host_pc, Some(0x1234));
        assert_eq!(samples[0].wall_time_ns, 12_000);
        assert_eq!(samples[0].cpu_time_ns, 9_000);
        assert_eq!(samples[0].guest_pcs, vec![0x30, 0x20]);
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
