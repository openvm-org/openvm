use std::borrow::Borrow;

use ax_stark_backend::{
    config::{StarkGenericConfig, Val},
    engine::StarkEngine,
    keygen::types::MultiStarkVerifyingKey,
    prover::types::Proof,
};
use ax_stark_sdk::{
    config::{baby_bear_poseidon2::BabyBearPoseidon2Engine, setup_tracing, FriParameters},
    engine::{ProofInputForTest, StarkFriEngine},
};
use axvm_instructions::{exe::AxVmExe, program::Program};
use p3_baby_bear::BabyBear;
use p3_field::{AbstractField, PrimeField32};

use crate::{
    arch::{ExitCode, PersistenceType, VirtualMachine, VmConfig, CONNECTOR_AIR_ID, MERKLE_AIR_ID},
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{merkle::MemoryMerklePvs, CHUNK},
    },
};

pub fn air_test(vm: VirtualMachine<BabyBear>, exe: impl Into<AxVmExe<BabyBear>>) {
    air_test_with_min_segments(vm, exe, vec![], 1);
}

pub fn air_test_with_min_segments(
    vm: VirtualMachine<BabyBear>,
    exe: impl Into<AxVmExe<BabyBear>>,
    input: Vec<Vec<BabyBear>>,
    min_segments: usize,
) {
    setup_tracing();

    let persistence_type = vm.config.memory_config.persistence_type;

    let engine = BabyBearPoseidon2Engine::new(FriParameters::standard_fast());
    let pk = vm.config.generate_pk(engine.keygen_builder());

    let result = vm.execute_and_generate(exe, input).unwrap();

    let proofs: Vec<Proof<_>> = result
        .per_segment
        .into_iter()
        .map(|proof_input| engine.prove(&pk, proof_input))
        .collect();

    assert!(proofs.len() >= min_segments);

    match persistence_type {
        PersistenceType::Volatile => {
            assert_eq!(proofs.len(), 1);
            engine
                .verify(&pk.get_vk(), &proofs.into_iter().next().unwrap())
                .expect("segment proof should verify");
        }
        PersistenceType::Persistent => {
            verify_segment_proofs(engine, &pk.get_vk(), &proofs);
        }
    }
}

pub fn verify_segment_proofs<SC: StarkGenericConfig>(
    engine: impl StarkEngine<SC>,
    vk: &MultiStarkVerifyingKey<SC>,
    proofs: &[Proof<SC>],
) {
    let mut prev_final_memory_root = None;
    let mut prev_final_pc = None;

    for (i, proof) in proofs.iter().enumerate() {
        engine
            .verify(vk, proof)
            .expect("segment proof should verify");

        // Check public values.
        for air_proof_data in proof.per_air.iter() {
            let pvs = &air_proof_data.public_values;
            let air_vk = &vk.per_air[air_proof_data.air_id];

            if air_proof_data.air_id == CONNECTOR_AIR_ID {
                let pvs: &VmConnectorPvs<_> = pvs.as_slice().borrow();

                if i != 0 {
                    // Check initial pc matches the previous final pc.
                    assert_eq!(pvs.initial_pc, prev_final_pc.unwrap());
                } else {
                    // TODO: Fetch initial pc from program
                }
                prev_final_pc = Some(pvs.final_pc);

                let expected_is_terminate = i == proofs.len() - 1;
                assert_eq!(
                    pvs.is_terminate,
                    Val::<SC>::from_bool(expected_is_terminate)
                );

                let expected_exit_code = if expected_is_terminate {
                    ExitCode::Success as u32
                } else {
                    DEFAULT_SUSPEND_EXIT_CODE
                };
                assert_eq!(
                    pvs.exit_code,
                    Val::<SC>::from_canonical_u32(expected_exit_code)
                );
            } else if air_proof_data.air_id == MERKLE_AIR_ID {
                let pvs: &MemoryMerklePvs<_, CHUNK> = pvs.as_slice().borrow();

                // Check that initial root matches the previous final root.
                if i != 0 {
                    assert_eq!(pvs.initial_root, prev_final_memory_root.unwrap());
                }
                prev_final_memory_root = Some(pvs.final_root);
            } else {
                assert_eq!(pvs.len(), 0);
                assert_eq!(air_vk.params.num_public_values, 0);
            }
        }
    }
}

/// Generates the VM STARK circuit, in the form of AIRs and traces, but does not
/// do any proving. Output is the payload of everything the prover needs.
///
/// The output AIRs and traces are sorted by height in descending order.
pub fn gen_vm_program_test_proof_input<SC: StarkGenericConfig>(
    program: Program<Val<SC>>,
    input_stream: Vec<Vec<Val<SC>>>,
    config: VmConfig,
) -> ProofInputForTest<SC>
where
    Val<SC>: PrimeField32,
{
    cfg_if::cfg_if! {
        if #[cfg(feature = "bench-metrics")] {
            // Run once with metrics collection enabled, which can improve runtime performance
            let mut config = config;
            config.collect_metrics = true;
            {
                let vm = VirtualMachine::<Val<SC>>::new(config.clone());
                vm.execute(program.clone(), input_stream.clone()).unwrap();
            }
            // Run again with metrics collection disabled and measure trace generation time
            config.collect_metrics = false;
            let start = std::time::Instant::now();
        }
    }

    let vm = VirtualMachine::<Val<SC>>::new(config);

    let mut result = vm.execute_and_generate(program, input_stream).unwrap();
    assert_eq!(
        result.per_segment.len(),
        1,
        "only proving one segment for now"
    );

    let result = result.per_segment.pop().unwrap();
    #[cfg(feature = "bench-metrics")]
    {
        metrics::gauge!("trace_gen_time_ms").set(start.elapsed().as_millis() as f64);
    }

    ProofInputForTest {
        per_air: result.into_air_proof_input_vec(),
    }
}
