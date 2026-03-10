// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { Halo2Verifier } from "./Halo2Verifier.sol";
import { IOpenVmHalo2Verifier } from "./interfaces/IOpenVmHalo2Verifier.sol";

type MemoryPointer is uint256;

/// @notice This contract provides a thin wrapper around the Halo2 verifier
/// outputted by `snark-verifier`, exposing a more user-friendly interface.
contract OpenVmHalo2Verifier is Halo2Verifier, IOpenVmHalo2Verifier {
    /// @dev Invalid public values length
    error InvalidPublicValuesLength(uint256 expected, uint256 actual);

    /// @dev Invalid proof data length
    error InvalidProofDataLength(uint256 expected, uint256 actual);

    /// @dev Proof verification failed
    error ProofVerificationFailed();

    /// @dev The length of the proof data, in bytes.
    uint256 private constant PROOF_DATA_LENGTH = (12 + 43) * 32;

    /// @dev The length of the public values, in bytes. This value is set by
    /// OpenVM and is guaranteed to be no larger than 8192.
    uint256 private constant PUBLIC_VALUES_LENGTH = {PUBLIC_VALUES_LENGTH};

    /// @dev The length of the full proof, in bytes
    uint256 private constant FULL_PROOF_LENGTH = (12 + 2 + PUBLIC_VALUES_LENGTH + 43) * 32;

    /// @dev The version of OpenVM that generated this verifier.
    string public constant OPENVM_VERSION = "{OPENVM_VERSION}";

    /// @notice A wrapper that constructs the proof into the right format for
    /// use with the `snark-verifier` verification.
    ///
    /// @dev The verifier expected proof format is:
    /// proof[..12 * 32]: KZG accumulator
    /// proof[12 * 32..13 * 32]: app exe commit
    /// proof[13 * 32..14 * 32]: app vm commit
    /// proof[14 * 32..(14 + PUBLIC_VALUES_LENGTH) * 32]: publicValues[0..PUBLIC_VALUES_LENGTH]
    /// proof[(14 + PUBLIC_VALUES_LENGTH) * 32..]: Proof Suffix
    ///
    /// @param publicValues The PVs revealed by the OpenVM guest program.
    /// @param proofData All components of the proof except the public values and
    /// app exe and vm commits. The expected format is:
    /// `abi.encodePacked(kzgAccumulator, proofSuffix)`
    /// @param appExeCommit The commitment to the OpenVM application executable whose execution
    /// is being verified.
    /// @param appVmCommit The commitment to the VM configuration.
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit, bytes32 appVmCommit) external view {
        if (publicValues.length != PUBLIC_VALUES_LENGTH) revert InvalidPublicValuesLength(PUBLIC_VALUES_LENGTH, publicValues.length);
        if (proofData.length != PROOF_DATA_LENGTH) revert InvalidProofDataLength(PROOF_DATA_LENGTH, proofData.length);

        // We will format the public values and construct the full proof payload
        // below.

        MemoryPointer proofPtr = _constructProof(publicValues, proofData, appExeCommit, appVmCommit);

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
    ///     abi.encodePacked(proofData[0:0x180], appExeCommit, appVmCommit, publicValuesPayload, proofData[0x180:]);
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
    /// proof. This pointer does not follow `bytes memory` semantics.
    function _constructProof(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit, bytes32 appVmCommit)
        internal
        pure
        returns (MemoryPointer proofPtr)
    {
        uint256 fullProofLength = FULL_PROOF_LENGTH;

        // The expected proof format using hex offsets:
        //
        // proof[..0x180]: KZG accumulator
        // proof[0x180..0x1a0]: app exe commit
        // proof[0x1a0..0x1c0]: app vm commit
        // proof[0x1c0..(0x1c0 + PUBLIC_VALUES_LENGTH * 32)]: publicValues[0..PUBLIC_VALUES_LENGTH]
        // proof[(0x1c0 + PUBLIC_VALUES_LENGTH * 32)..]: Proof Suffix

        /// @solidity memory-safe-assembly
        assembly {
            proofPtr := mload(0x40)
            // Allocate the memory as a safety measure.
            mstore(0x40, add(proofPtr, fullProofLength))

            // Copy the KZG accumulator (length 0x180) into the beginning of
            // the memory buffer
            calldatacopy(proofPtr, proofData.offset, 0x180)

            // Copy the App Exe Commit and App Vm Commit into the memory buffer
            mstore(add(proofPtr, 0x180), appExeCommit)
            mstore(add(proofPtr, 0x1a0), appVmCommit)

            // Copy the Proof Suffix (length 43 * 32 = 0x560) into the
            // end of the memory buffer, leaving PUBLIC_VALUES_LENGTH words in
            // between for the publicValuesPayload.
            //
            // Begin copying from the end of the KZG accumulator in the
            // calldata buffer (0x180)
            let proofSuffixOffset := add(0x1c0, shl(5, PUBLIC_VALUES_LENGTH))
            calldatacopy(add(proofPtr, proofSuffixOffset), add(proofData.offset, 0x180), 0x560)

            // Copy each byte of the public values into the proof. It copies the
            // most significant bytes of public values first.
            let publicValuesMemOffset := add(add(proofPtr, 0x1c0), 0x1f)
            for { let i := 0 } iszero(eq(i, PUBLIC_VALUES_LENGTH)) { i := add(i, 1) } {
                calldatacopy(add(publicValuesMemOffset, shl(5, i)), add(publicValues.offset, i), 0x01)
            }
        }
    }
}
