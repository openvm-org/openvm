use std::sync::Arc;

use once_cell::sync::Lazy;
use prometheus_client::{
    encoding::{text::encode, EncodeLabelSet, EncodeLabelValue},
    metrics::{family::Family, gauge::Gauge},
    registry::Registry,
};
use sysinfo::System;

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelSet)]
struct Label {
    method: Method,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq, EncodeLabelValue)]
pub enum Method {
    Build,
    Transpile,
    Execute,
    Commit,
    KeyGen,
    Proof,
    Verify,
    AggVerify,
    AggKeyGen,
    EvmProof,
    SnarkVerifierContract,
    VerifyEvmProof,
}

type MemoryUsage = Family<Label, Gauge>;

pub struct Profiler {
    registry: Registry,
    memory_gauge: MemoryUsage,
}

impl Profiler {
    pub fn new() -> Self {
        let mut registry = Registry::default();
        let memory_gauge = MemoryUsage::default();
        registry.register(
            "memory usage",
            "memory used in process",
            memory_gauge.clone(),
        );
        Self {
            registry,
            memory_gauge,
        }
    }

    pub fn update_memory_usage(&self, method: Method) {
        let mut system = System::new_all();
        system.refresh_memory(); // Refresh memory stats
        let used_memory = system.used_memory(); // Memory used in bytes
                                                // println!("{:?}", used_memory);
        self.memory_gauge
            .get_or_create(&Label { method })
            .set(used_memory as i64);
    }

    pub fn print_metrics(&self) {
        // Print memory usage
        let mut buffer = String::new();
        encode(&mut buffer, &self.registry).unwrap();
        println!("{}", buffer);
    }
}

impl Drop for Profiler {
    fn drop(&mut self) {
        self.print_metrics();
    }
}
