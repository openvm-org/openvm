// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { Halo2Verifier } from "./Halo2Verifier.sol";
import { IOpenVmHalo2Verifier } from "./interfaces/IOpenVmHalo2Verifier.sol";

type MemoryPointer is uint256;

// BN254 scalar field modulus (Fr), as specified in EIP-197:
// https://eips.ethereum.org/EIPS/eip-197
uint256 constant BN254_SCALAR_MODULUS = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001;

/// @notice This contract provides a thin wrapper around the Halo2 verifier
/// outputted by `snark-verifier`, exposing a more user-friendly interface.
contract OpenVmHalo2Verifier is Halo2Verifier, IOpenVmHalo2Verifier {
    /// @dev Invalid public-values byte length
    error InvalidPublicValuesLength(uint256 expected, uint256 actual);

    /// @dev Invalid number of revealed u64 values
    error InvalidPublicValuesCount(uint256 max, uint256 actual);

    /// @dev Invalid proof data length
    error InvalidProofDataLength(uint256 expected, uint256 actual);

    /// @dev Invalid app executable commitment
    error InvalidAppExeCommit(bytes32 actual);

    /// @dev Invalid app VM commitment
    error InvalidAppVmCommit(bytes32 actual);

    /// @dev Proof verification failed
    error ProofVerificationFailed();

    /// @dev The length of the proof data, in bytes.
    uint256 private constant PROOF_DATA_LENGTH = (12 + 43) * 32;

    /// @dev Number of fixed-capacity u16 public-value cells. This value is set
    /// by OpenVM and is guaranteed to be no larger than 8192.
    uint256 private constant PUBLIC_VALUE_CELLS = {PUBLIC_VALUE_CELLS};

    /// @dev Each public-value cell is encoded as two little-endian bytes.
    uint256 private constant PUBLIC_VALUES_LENGTH = PUBLIC_VALUE_CELLS * 2;

    /// @dev Each revealed u64 occupies four u16 cells.
    uint256 private constant PUBLIC_VALUES_CAPACITY = PUBLIC_VALUE_CELLS / 4;

    /// @dev The length of the full proof, in bytes
    uint256 private constant FULL_PROOF_LENGTH = (12 + 2 + 1 + PUBLIC_VALUE_CELLS + 43) * 32;

    /// @dev The version of OpenVM that generated this verifier.
    string private constant OPENVM_VERSION = "{OPENVM_VERSION}";

    /// @notice A wrapper that constructs the proof into the right format for
    /// use with the `snark-verifier` verification.
    ///
    /// @dev The verifier expected proof format is:
    /// proof[..12 * 32]: KZG accumulator
    /// proof[12 * 32..13 * 32]: app exe commit
    /// proof[13 * 32..14 * 32]: app vm commit
    /// proof[14 * 32..15 * 32]: number of revealed u64 public values
    /// proof[15 * 32..(15 + PUBLIC_VALUE_CELLS) * 32]: u16 public-value cells
    /// proof[(15 + PUBLIC_VALUE_CELLS) * 32..]: Proof Suffix
    ///
    /// @param publicValuesCount Number of u64 values revealed by the guest.
    /// @param publicValues Fixed-capacity u16 cells encoded as little-endian byte pairs.
    /// @param proofData All components of the proof except the public values and
    /// app exe and vm commits. The expected format is:
    /// `abi.encodePacked(kzgAccumulator, proofSuffix)`
    /// @param appExeCommit The commitment to the OpenVM application executable whose execution
    /// is being verified.
    /// @param appVmCommit The commitment to the VM configuration.
    function verify(
        uint32 publicValuesCount,
        bytes calldata publicValues,
        bytes calldata proofData,
        bytes32 appExeCommit,
        bytes32 appVmCommit
    ) external view {
        if (publicValuesCount > PUBLIC_VALUES_CAPACITY) {
            revert InvalidPublicValuesCount(PUBLIC_VALUES_CAPACITY, publicValuesCount);
        }
        if (publicValues.length != PUBLIC_VALUES_LENGTH) revert InvalidPublicValuesLength(PUBLIC_VALUES_LENGTH, publicValues.length);
        if (proofData.length != PROOF_DATA_LENGTH) revert InvalidProofDataLength(PROOF_DATA_LENGTH, proofData.length);
        if (uint256(appExeCommit) >= BN254_SCALAR_MODULUS) revert InvalidAppExeCommit(appExeCommit);
        if (uint256(appVmCommit) >= BN254_SCALAR_MODULUS) revert InvalidAppVmCommit(appVmCommit);

        // Other than the fallback() in `Halo2Verifier`, there is only one
        // function selector on the external ABI: `verify(..)`, which has
        // selector 0x5ee4cdd6. If `proofData` ever began with 0x5ee4cdd6, this
        // function would be called again instead of hitting the fallback. 
        //
        // If a valid proof ever began with 0x5ee4cdd6, it would fail to verify.
        // However, `snark-verifier`'s proof structure guarantees that the first
        // 12 words of the proof are the KZG accumulator limbs. Each limb holds
        // 88 bits, so the first 4 bytes of the proof will always be
        // 0x00000000.
        //
        // As an extra layer of protection, we assert that the first 4 bytes of
        // `proofData` are 0x00000000 before self-calling into the fallback
        // verifier.
        assert(bytes4(proofData[0:4]) == bytes4(0x00000000));

        // We will format the public values and construct the full proof payload
        // below.

        MemoryPointer proofPtr =
            _constructProof(publicValuesCount, publicValues, proofData, appExeCommit, appVmCommit);

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
    ///     abi.encodePacked(proofData[0:0x180], appExeCommit, appVmCommit,
    ///         bytes32(uint256(publicValuesCount)), publicValuesPayload, proofData[0x180:]);
    /// ```
    //
    /// where `publicValuesPayload` has each little-endian u16 pair in
    /// `publicValues` converted to its own canonical `bytes32` field word.
    ///
    /// This function does not clean the memory it allocates. Since it is the
    /// only memory write that occurs in the call frame, we know that
    /// the memory region cannot have been dirtied.
    ///
    /// @return proofPtr Memory pointer to the beginning of the constructed
    /// proof. This pointer does not follow `bytes memory` semantics.
    function _constructProof(
        uint32 publicValuesCount,
        bytes calldata publicValues,
        bytes calldata proofData,
        bytes32 appExeCommit,
        bytes32 appVmCommit
    )
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
        // proof[0x1c0..0x1e0]: number of revealed u64 public values
        // proof[0x1e0..(0x1e0 + PUBLIC_VALUE_CELLS * 32)]: u16 public-value cells
        // proof[(0x1e0 + PUBLIC_VALUE_CELLS * 32)..]: Proof Suffix

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
            mstore(add(proofPtr, 0x1c0), publicValuesCount)

            // Copy the Proof Suffix (length 43 * 32 = 0x560) into the
            // end of the memory buffer, leaving one count word and
            // PUBLIC_VALUE_CELLS words in between.
            //
            // Begin copying from the end of the KZG accumulator in the
            // calldata buffer (0x180)
            let proofSuffixOffset := add(0x1e0, shl(5, PUBLIC_VALUE_CELLS))
            calldatacopy(add(proofPtr, proofSuffixOffset), add(proofData.offset, 0x180), 0x560)

            // Decode every little-endian byte pair as one canonical u16 field.
            let publicValuesMemOffset := add(proofPtr, 0x1e0)
            for { let i := 0 } iszero(eq(i, PUBLIC_VALUE_CELLS)) { i := add(i, 1) } {
                let input := calldataload(add(publicValues.offset, shl(1, i)))
                let value := or(byte(0, input), shl(8, byte(1, input)))
                mstore(add(publicValuesMemOffset, shl(5, i)), value)
            }
        }
    }
}
