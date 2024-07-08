use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufRead, Write},
};

use color_eyre::eyre::Result;

const TIME_PREFIX: &str = "time.busy=";

pub fn clear_tracing_log(file_path: &str) -> Result<()> {
    let mut file = File::create(file_path)?;
    file.write_all(b"")?;
    Ok(())
}

pub fn filter_and_extract_time_busy(
    file_path: &str,
    filter_values: &[&str],
) -> Result<HashMap<String, String>> {
    let mut results: HashMap<String, String> = HashMap::new();

    if let Ok(file) = File::open(file_path) {
        for line in io::BufReader::new(file).lines() {
            let line = line.unwrap();
            for &val in filter_values {
                if line.contains(val) {
                    if let Some(start) = line.find(TIME_PREFIX) {
                        let time_busy_start = start + TIME_PREFIX.len();
                        if let Some(end) = line[time_busy_start..].find(' ') {
                            let time_busy =
                                line[time_busy_start..time_busy_start + end].to_string();
                            results.insert(val.to_string(), time_busy);
                        }
                    }
                }
            }
        }
    }

    Ok(results)
}
