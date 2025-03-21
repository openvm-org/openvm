use openvm_native_compiler::ir::DIGEST_SIZE;
use openvm_stark_backend::{
    config::{Com, PcsProof, StarkGenericConfig},
    p3_commit::OpenedValues,
    p3_field::{FieldExtensionAlgebra, PrimeField32},
    proof::{AdjacentOpenedValues, OpeningProof, Proof},
};

use super::{F, SC}; // BabyBearPoseidon2Config

type Challenge = <SC as StarkGenericConfig>::Challenge;

// We need to know:
// - Pcs is TwoAdicFriPcs
// - Com<SC>: Into<[F; 8]>
pub fn encode_proof(proof: &Proof<SC>) -> Vec<u32> {
    let mut encoded = Vec::new();
    // main
    encoded.push(proof.commitments.main_trace.len().try_into().unwrap());
    for &com in &proof.commitments.main_trace {
        encoded.extend(encode_commitment(com));
    }
    // after challenge
    encoded.push(proof.commitments.after_challenge.len().try_into().unwrap());
    for &com in &proof.commitments.after_challenge {
        encoded.extend(encode_commitment(com));
    }
    // quotient
    encoded.extend(encode_commitment(proof.commitments.quotient));

    // Encode OpeningProof
    encoded.extend(encode_opening_proof(&proof.opening));

    // Encode per_air data
    encoded.push(proof.per_air.len().try_into().unwrap());
    for air_data in &proof.per_air {
        encoded.extend(encode_air_proof_data(air_data));
    }

    // Encode rap_phase_seq_proof if it exists
    encoded.push(if proof.rap_phase_seq_proof.is_some() {
        1
    } else {
        0
    });
    if let Some(rap_proof) = &proof.rap_phase_seq_proof {
        encoded.extend(encode_rap_phase_seq_proof(rap_proof));
    }

    encoded
}

fn encode_opening_proof(opening: &OpeningProof<PcsProof<SC>, Challenge>) -> Vec<u32> {
    let mut encoded = Vec::new();

    // Encode PCS proof (this would depend on the PcsProof structure)
    encoded.extend(encode_pcs_proof(&opening.proof));

    // Encode OpenedValues
    encoded.extend(encode_opened_values(&opening.values));

    encoded
}

fn encode_opened_values(values: &OpenedValues<SC::Challenge>) -> Vec<u32> {
    let mut encoded = Vec::new();

    // Encode preprocessed values
    encoded.push(values.preprocessed.len().try_into().unwrap());
    for adjacent in &values.preprocessed {
        encoded.extend(encode_adjacent_opened_values(adjacent));
    }

    // Encode main values
    encoded.push(values.main.len().try_into().unwrap());
    for matrices in &values.main {
        encoded.push(matrices.len().try_into().unwrap());
        for adjacent in matrices {
            encoded.extend(encode_adjacent_opened_values(adjacent));
        }
    }

    // Encode after_challenge values
    encoded.push(values.after_challenge.len().try_into().unwrap());
    for matrices in &values.after_challenge {
        encoded.push(matrices.len().try_into().unwrap());
        for adjacent in matrices {
            encoded.extend(encode_adjacent_opened_values(adjacent));
        }
    }

    // Encode quotient values
    encoded.push(values.quotient.len().try_into().unwrap());
    for rap in &values.quotient {
        encoded.push(rap.len().try_into().unwrap());
        for chunk in rap {
            encoded.push(chunk.len().try_into().unwrap());
            for &val in chunk {
                encoded.push(encode_challenge(val));
            }
        }
    }

    encoded
}

fn encode_adjacent_opened_values(adjacent: &AdjacentOpenedValues<Challenge>) -> Vec<u32> {
    let mut encoded = Vec::new();

    // Encode local values
    encoded.push(adjacent.local.len().try_into().unwrap());
    for &val in &adjacent.local {
        encoded.push(encode_challenge(val));
    }

    // Encode next values
    encoded.push(adjacent.next.len().try_into().unwrap());
    for &val in &adjacent.next {
        encoded.push(encode_challenge(val));
    }

    encoded
}

fn encode_air_proof_data(data: &AirProofData<Val<SC>, SC::Challenge>) -> Vec<u32> {
    let mut encoded = Vec::new();

    // Encode air_id
    encoded.push(data.air_id.try_into().unwrap());

    // Encode degree
    encoded.push(data.degree.try_into().unwrap());

    // Encode exposed_values_after_challenge
    encoded.push(
        data.exposed_values_after_challenge
            .len()
            .try_into()
            .unwrap(),
    );
    for phase in &data.exposed_values_after_challenge {
        encoded.push(phase.len().try_into().unwrap());
        for &val in phase {
            encoded.push(encode_challenge(val));
        }
    }

    // Encode public_values
    encoded.push(data.public_values.len().try_into().unwrap());
    for &val in &data.public_values {
        encoded.push(encode_val(val));
    }

    encoded
}

// Helper function to encode PcsProof (depends on actual structure)
fn encode_pcs_proof(proof: &PcsProof<SC>) -> Vec<u32> {
    // Implementation would depend on the structure of PcsProof<SC>
    // For example, if it contains FRI proofs, commitments, etc.
    // This is a placeholder
    let mut encoded = Vec::new();

    // You would need to implement this based on PcsProof structure
    // ...

    encoded
}

// Function to encode RapPhaseSeqPartialProof
fn encode_rap_phase_seq_proof(proof: &RapPhaseSeqPartialProof<SC>) -> Vec<u32> {
    // Implementation would depend on the structure of RapPhaseSeqPartialProof<SC>
    // This is a placeholder
    let mut encoded = Vec::new();

    // You would need to implement this based on RapPhaseSeqPartialProof structure
    // ...

    encoded
}

fn encode_commitment(com: Com<SC>) -> [u32; DIGEST_SIZE] {
    let com_array: [F; DIGEST_SIZE] = com.into();
    com_array.map(encode_val)
}

// Helper function to encode Challenge type
fn encode_challenge(challenge: <SC as StarkGenericConfig>::Challenge) -> Vec<u32> {
    let base_slice: &[F] = challenge.as_base_slice();
    base_slice.iter().copied().map(encode_val).collect()
}

fn encode_val<F: PrimeField32>(val: F) -> u32 {
    val.as_canonical_u32()
}
