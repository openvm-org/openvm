use std::{env, error::Error, fmt::Write as _, path::Path};

use bitcode::serialize;
use openvm_sdk::{fs::read_object_from_file, keygen::AppVerifyingKey};

fn main() -> Result<(), Box<dyn Error>> {
    let path = env::args().nth(1).expect("usage: vk_dump <app.vk>");
    let vk = read_object_from_file::<AppVerifyingKey, _>(Path::new(&path))?;

    println!("vk path: {path}");
    print_hash("vk", &vk)?;

    println!("\n=== fri_params ===");
    println!("{:?}", vk.fri_params);

    println!("\n=== memory_dimensions ===");
    println!("{:?}", vk.memory_dimensions);

    println!("\n=== vk::pre_hash ===");
    println!("pre_hash: {:02x?}", serialize(&vk.vk.pre_hash)?.as_slice());

    println!("\n=== vk::inner ===");
    let inner_bytes = serialize(&vk.vk.inner)?;
    print_hash_bytes("inner", inner_bytes.as_slice());

    println!("\n=== trace_height_constraints ===");
    print_hash_bytes(
        "trace_height_constraints",
        serialize(&vk.vk.inner.trace_height_constraints)?.as_slice(),
    );

    println!("\n=== per_air ===");
    for (idx, air) in vk.vk.inner.per_air.iter().enumerate() {
        println!("air #{idx}");
        println!(
            "  width main={:?} common={} cached={:?} after={:?} pubs={} quotient={} rap={:?}",
            air.params.width.main_widths(),
            air.params.width.common_main,
            air.params.width.cached_mains,
            air.params.width.after_challenge,
            air.params.num_public_values,
            air.quotient_degree,
            air.rap_phase_seq_kind,
        );
        println!(
            "  num_challenges_per_phase={:?} num_exposed_after_challenge={:?}",
            air.params.num_challenges_to_sample, air.params.num_exposed_values_after_challenge,
        );
        print_hash_bytes("  air", serialize(air)?.as_slice());
        print_hash_bytes(
            "  symbolic",
            serialize(&air.symbolic_constraints)?.as_slice(),
        );
        let num_nodes = air.symbolic_constraints.constraints.nodes.len();
        let constraint_idx = &air.symbolic_constraints.constraints.constraint_idx;
        println!(
            "  symbolic nodes={} constraint_idx={:?}",
            num_nodes, constraint_idx
        );
    }

    Ok(())
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn print_hash(label: &str, vk: &AppVerifyingKey) -> Result<(), Box<dyn Error>> {
    let bytes = serialize(vk)?;
    print_hash_bytes(label, &bytes);
    Ok(())
}

fn print_hash_bytes(label: &str, bytes: &[u8]) {
    let digest = fnv1a(bytes);
    let mut prefix = String::new();
    for b in bytes.iter().take(16) {
        let _ = write!(&mut prefix, "{b:02x}");
    }
    println!(
        "{label}: len={} fnv=0x{digest:016x} first16={prefix}",
        bytes.len()
    );
}
