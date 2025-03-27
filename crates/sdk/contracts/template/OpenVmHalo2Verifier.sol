// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { Halo2Verifier } from "./Halo2Verifier.sol";
import { IOpenVmHalo2Verifier } from "./interfaces/IOpenVmHalo2Verifier.sol";

type MemoryPointer is uint256;

/// @notice This contract provides a thin wrapper around the Halo2 verifier
/// outputted by `snark-verifier`, exposing a more user-friendly interface.
contract OpenVmHalo2Verifier is Halo2Verifier, IOpenVmHalo2Verifier {
    /// @dev Invalid proof data length
    error InvalidProofDataLength();

    /// @dev Invalid public values length
    error InvalidPublicValuesLength();

    /// @dev Proof verification failed
    error ProofVerificationFailed();

    /// @dev The length of the proof data, in bytes
    uint256 private constant PROOF_DATA_LENGTH = (12 + 43) * 32;

    // /// @dev The length of the public values, in bytes. This value is set by OpenVM.
    // uint256 private constant PUBLIC_VALUES_LENGTH = 32;

    /// @dev The length of the full proof, in bytes
    uint256 private constant FULL_PROOF_LENGTH = (12 + 2 + PUBLIC_VALUES_LENGTH + 43) * 32;

    /// @dev The leaf verifier commitment. This value is set by OpenVM.
    bytes32 public constant LEAF_EXE_COMMIT =
        bytes32(0x0071628bff0dcb64201f77ff5c7d869c7073b842e3dadf9e618e8673ef671bfd);

    /// @dev The version of OpenVM that generated the proof.
    string public constant OPENVM_VERSION = "v1.0.0";

    /// @notice A wrapper that constructs the proof into the right format for
    /// use with the `snark-verifier` verification.
    ///
    /// @dev This function assumes that `publicValues` encodes one `bytes32`
    /// hash which is the hash of the public values.
    ///
    /// The verifier expected proof format is:
    /// proof[..12 * 32]: KZG accumulators
    /// proof[12 * 32..13 * 32]: app exe commit
    /// proof[13 * 32..14 * 32]: leaf exe commit
    /// proof[14 * 32..(14 + PUBLIC_VALUES_LENGTH) * 32]: publicValues[0..PUBLIC_VALUES_LENGTH]
    /// proof[(14 + PUBLIC_VALUES_LENGTH) * 32..]: Public Values Suffix
    ///
    /// @param publicValues The PVs revealed by the OpenVM guest program.
    /// @param proofData All components of the proof except the public values,
    /// leaf and app exe commits. The expected format is:
    /// `abi.encodePacked(kzgAccumulators, proofSuffix)`
    /// @param appExeCommit The commitment to the OpenVM application executable whose execution
    /// is being verified.
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit) external view {
        if (publicValues.length != PUBLIC_VALUES_LENGTH) revert InvalidPublicValuesLength();
        if (proofData.length != PROOF_DATA_LENGTH) revert InvalidProofDataLength();

        // We will format the public values and construct the full proof payload
        // below.

        MemoryPointer proofPtr = _constructProof(publicValues, proofData, appExeCommit);

        uint256 fullProofLength = FULL_PROOF_LENGTH;

        /// @solidity memory-safe-assembly
        assembly {
            // Self-call using the proof as calldata
            if iszero(staticcall(gas(), address(), proofPtr, fullProofLength, 0, 0)) {
                mstore(0x00, 0xd611c318) // ProofVerificationFailed()
                revert(0x1c, 0x04)
            }
        }
    }

    /// @dev The assembly code should perform the same function as the following
    /// solidity code:
    //
    /// ```solidity
    /// bytes memory proof =
    ///     abi.encodePacked(proofData[0:0x180], appExeCommit, leafExeCommit, publicValuesPayload, proofData[0x180:]);
    /// ```
    //
    /// where `publicValuesPayload` is a memory payload with each byte in
    /// `publicValues` separated into its own `bytes32` word.
    ///
    /// This function does not clean the memory it allocates. Since it is the
    /// only memory write that occurs in the call frame, we know that
    /// the memory region cannot have been dirtied.
    ///
    /// @return proofPtr Memory pointer to the beginning of the constructed
    /// proof.
    function _constructProof(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit)
        internal
        pure
        returns (MemoryPointer proofPtr)
    {
        uint256 fullProofLength = FULL_PROOF_LENGTH;
        bytes32 leafExeCommit = LEAF_EXE_COMMIT;

        // The expected proof format using hex offsets:
        //
        // proof[..0x180]: KZG accumulators
        // proof[0x180..0x1a0]: app exe commit
        // proof[0x1a0..0x1c0]: leaf exe commit
        // proof[0x1c0..(0x1c0 + PUBLIC_VALUES_LENGTH * 32)]: publicValues[0..PUBLIC_VALUES_LENGTH]
        // proof[(0x1c0 + PUBLIC_VALUES_LENGTH * 32)..]: Public Values Suffix

        /// @solidity memory-safe-assembly
        assembly {
            proofPtr := mload(0x40)
            // Allocate the memory as a safety measure.
            mstore(0x40, add(proofPtr, fullProofLength))

            // Copy the KZG accumulators (length 0x180) into the beginning of
            // the memory buffer
            calldatacopy(proofPtr, proofData.offset, 0x180)

            // Copy the App Exe Commit and Leaf Exe Commit into the memory buffer
            mstore(add(proofPtr, 0x180), appExeCommit)
            mstore(add(proofPtr, 0x1a0), leafExeCommit)

            // Copy the Public Values Suffix (length 43 * 32 = 0x560) into the
            // end of the memory buffer, leaving PUBLIC_VALUES_LENGTH words in
            // between for the publicValuesPayload.
            //
            // Begin copying from the end of the KZG accumulators in the
            // calldata buffer (0x180)
            let suffixProofOffset := add(0x1c0, shl(5, PUBLIC_VALUES_LENGTH))
            calldatacopy(add(proofPtr, suffixProofOffset), add(proofData.offset, 0x180), 0x560)

            // Copy each byte of the public values into the proof. It copies the
            // most significant bytes of public values first.
            let publicValuesMemOffset := add(add(proofPtr, 0x1c0), 0x1f)
            for { let i := 0 } iszero(eq(i, PUBLIC_VALUES_LENGTH)) { i := add(i, 1) } {
                calldatacopy(add(publicValuesMemOffset, shl(5, i)), add(publicValues.offset, i), 0x01)
            }
        }
    }
}
