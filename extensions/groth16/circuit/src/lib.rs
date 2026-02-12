use openvm_circuit::chip::Chip;
use openvm_circuit::arch::ExecutionState;
use openvm_instructions::instruction::Instruction;
use openvm_groth16_guest::{GROTH16_VERIFY_OPCODE, Groth16Proof, Groth16VerifyingKey};
use openvm_algebra_circuit::bn254::{Bn254PairingChip, Bn254G1Point, Bn254G2Point};

pub struct Groth16Chip {
    pub pairing_chip: Bn254PairingChip,
}

impl Chip for Groth16Chip {
    fn execute(&self, execution_state: &mut ExecutionState, instruction: Instruction) {
        if instruction.opcode != GROTH16_VERIFY_OPCODE {
            return;
        }

        // Operands: [proof_ptr, vk_ptr, pub_input_ptr, pub_input_len]
        let proof_ptr = instruction.operands[0];
        let vk_ptr = instruction.operands[1];
        let pub_ptr = instruction.operands[2];
        let pub_len = instruction.operands[3] as usize;

        // Read Proof from Memory
        let proof: Groth16Proof = execution_state.read_serialized_struct(proof_ptr);
        let vk: Groth16VerifyingKey = execution_state.read_serialized_struct(vk_ptr);

        // Process Public Inputs: Compute the "Sum" point in G1
        // Sum = VK.IC[0] + \sum_{i=0}^{n-1} (pub_input[i] * VK.IC[i+1])
        let public_input_sum = self.compute_public_input_sum(execution_state, pub_ptr, pub_len);

        // Queue the Pairing Equation:
        // e(A, B) == e(alpha, beta) * e(Sum, gamma) * e(C, delta)
        // Which is equivalent to checking if the product of pairings equals 1:
        // e(A, B) * e(alpha, -beta) * e(Sum, -gamma) * e(C, -delta) == 1

        // We add the four pairing pairs to the pairing chip's accumulator
        let mut pairing_request = self.pairing_chip.new_request();

        pairing_request.add(proof.a, proof.b);                    // e(A, B)
        pairing_request.add(vk.alpha_g1, vk.beta_g2.negate());    // e(alpha, -beta)
        pairing_request.add(public_input_sum, vk.gamma_g2.negate()); // e(Sum, -gamma)
        pairing_request.add(proof.c, vk.delta_g2.negate());       // e(C, -delta)

        // The PairingChip will enforce the multi-pairing check at the end of the segment
        self.pairing_chip.enforce_pairing_check(execution_state, pairing_request);
    }
}

impl Groth16Chip {
    fn compute_public_input_sum(
        &self,
        state: &mut ExecutionState,
        pub_ptr: u32,
        pub_len: usize,
        vk: &Groth16VerifyingKey
    ) -> Bn254G1Point {
        // Get the reference to the IC[0] point (the "accumulator" point)
        // In Groth16, VK.ic[0] is part of the verification check.
        let mut sum_point = vk.ic[0];

        // Iterate through public inputs and multiply them by IC[i+1]
        for i in 0..pub_len {
            // Read the i-th public input scalar from guest memory
            let scalar = state.read_memory(pub_ptr + (i * 4) as u32);

            // Get the corresponding G1 point from the VK (IC array)
            let ic_point = vk.ic[i + 1];

            // Perform ECC Scalar Multiplication: (scalar * ic_point)
            // This interacts with the underlying ZK circuit for ECC
            let term = self.pairing_chip.ecc_chip.scalar_mul(state, scalar, ic_point);

            // Add the result to the accumulator: sum = sum + term
            sum_point = self.pairing_chip.ecc_chip.add(state, sum_point, term);
        }

        sum_point
    }
}
