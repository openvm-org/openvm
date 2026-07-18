use std::{
    collections::BTreeSet,
    ffi::OsString,
    fs,
    path::Path,
    time::{Duration, SystemTime},
};

use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use eyre::{bail, eyre, Context, Result};
use flate2::{write::GzEncoder, Compression};
use fxprof_processed_profile::{
    CategoryHandle, CpuDelta, Frame, FrameFlags, FrameInfo, Profile, SamplingInterval, Timestamp,
};
use openvm_circuit::arch::rvr::{default_addr2line_cmd, GuestDebugMap};
use reqwest::{blocking::Client, header::ACCEPT};
use serde_json::Value;

const PROFILE_PATH_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE";
const PROFILE_HZ_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE_HZ";
const PROFILE_FORMAT_ENV: &str = "OPENVM_RVR_GUEST_CALL_PROFILE_FORMAT";
const DEFAULT_UPLOAD_URL: &str = "https://api.profiler.firefox.com/compressed-store";
const FIREFOX_ACCEPT: &str = "application/vnd.firefox-profiler+json;version=1.0";

/// Temporarily enables the low-overhead sampler owned by the RVR executor.
pub struct GuestProfileGuard {
    previous: [(OsString, Option<OsString>); 3],
}

impl GuestProfileGuard {
    pub fn start(path: &Path, sample_hz: u32) -> Self {
        let previous = [
            (PROFILE_PATH_ENV.into(), std::env::var_os(PROFILE_PATH_ENV)),
            (PROFILE_HZ_ENV.into(), std::env::var_os(PROFILE_HZ_ENV)),
            (
                PROFILE_FORMAT_ENV.into(),
                std::env::var_os(PROFILE_FORMAT_ENV),
            ),
        ];
        // The CLI performs a single execution. No other thread reads these
        // private RVR configuration variables before the guard is dropped.
        unsafe {
            std::env::set_var(PROFILE_PATH_ENV, path);
            std::env::set_var(PROFILE_HZ_ENV, sample_hz.to_string());
            std::env::set_var(PROFILE_FORMAT_ENV, "raw");
        }
        Self { previous }
    }
}

impl Drop for GuestProfileGuard {
    fn drop(&mut self) {
        for (key, value) in &self.previous {
            // See the safety note in `start`.
            unsafe {
                if let Some(value) = value {
                    std::env::set_var(key, value);
                } else {
                    std::env::remove_var(key);
                }
            }
        }
    }
}

pub fn create_upload_and_optionally_save(
    raw_profile_path: &Path,
    guest_elf_path: &Path,
    sample_hz: u32,
    output_path: Option<&Path>,
) -> Result<String> {
    let samples = parse_raw_samples(raw_profile_path)?;
    let profile = build_firefox_profile(&samples, guest_elf_path, sample_hz)?;
    let compressed = compress_profile(&profile)?;

    if let Some(output_path) = output_path {
        fs::write(output_path, &compressed).with_context(|| {
            format!(
                "failed to write Firefox profile to {}",
                output_path.display()
            )
        })?;
        eprintln!(
            "[openvm] Saved execution profile to {}",
            output_path.display()
        );
    }

    upload_profile(&compressed)
}

fn parse_raw_samples(path: &Path) -> Result<Vec<Vec<u32>>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read RVR samples from {}", path.display()))?;
    let mut samples = Vec::new();
    for (line_index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let stack = line
            .split(';')
            .map(|value| {
                let value = value.trim().strip_prefix("0x").unwrap_or(value.trim());
                u32::from_str_radix(value, 16)
                    .with_context(|| format!("invalid guest PC on profile line {}", line_index + 1))
            })
            .collect::<Result<Vec<_>>>()?;
        if !stack.is_empty() {
            samples.push(stack);
        }
    }
    if samples.is_empty() {
        bail!(
            "RVR execution completed before any profile samples were captured; try a larger workload or a higher --rate"
        );
    }
    Ok(samples)
}

fn build_firefox_profile(
    samples: &[Vec<u32>],
    guest_elf_path: &Path,
    sample_hz: u32,
) -> Result<Profile> {
    let lookup_pcs = samples
        .iter()
        .flat_map(|stack| {
            stack.iter().enumerate().map(|(index, &pc)| {
                if index + 1 == stack.len() {
                    pc
                } else {
                    pc.saturating_sub(1)
                }
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

    // ITIMER_PROF advances only while the process is consuming CPU. The raw
    // stack format does not carry clocks, so model every interval after the
    // first as CPU time. This keeps Firefox's activity graph consistent with
    // the synthetic sample timestamps instead of reporting zero CPU usage.
    let interval_ms = 1_000.0 / f64::from(sample_hz);
    for (sample_index, stack) in samples.iter().enumerate() {
        let frames = stack
            .iter()
            .enumerate()
            .map(|(frame_index, &pc)| {
                let lookup_pc = if frame_index + 1 == stack.len() {
                    pc
                } else {
                    pc.saturating_sub(1)
                };
                let label = debug_map
                    .get(lookup_pc)
                    .filter(|location| !location.function.is_empty())
                    .map(|location| location.function.clone())
                    .unwrap_or_else(|| format!("0x{pc:08x}"));
                FrameInfo {
                    frame: Frame::Label(profile.intern_string(&label)),
                    category_pair: CategoryHandle::OTHER.into(),
                    flags: FrameFlags::empty(),
                }
            })
            .collect::<Vec<_>>();
        let stack = profile.intern_stack_frames(thread, frames.into_iter());
        profile.add_sample(
            thread,
            Timestamp::from_millis_since_reference(sample_index as f64 * interval_ms),
            stack,
            if sample_index == 0 {
                CpuDelta::ZERO
            } else {
                CpuDelta::from_millis(interval_ms)
            },
            1,
        );
    }
    let end = Timestamp::from_millis_since_reference(
        samples.len().saturating_sub(1) as f64 * interval_ms,
    );
    profile.set_process_end_time(process, end);
    profile.set_thread_end_time(thread, end);
    Ok(profile)
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
    let upload_url = std::env::var("FIREFOX_PROFILER_API_URL")
        .unwrap_or_else(|_| DEFAULT_UPLOAD_URL.to_string());
    upload_profile_blocking_to(&upload_url, compressed_profile)
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
    fn parses_ordered_raw_stacks() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("samples.raw");
        fs::write(&path, "0x30;0x20;0x10\n0x40\n").unwrap();
        assert_eq!(
            parse_raw_samples(&path).unwrap(),
            vec![vec![0x30, 0x20, 0x10], vec![0x40]]
        );
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
