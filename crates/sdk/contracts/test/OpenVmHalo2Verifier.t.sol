// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { LibString } from "./helpers/LibString.sol";
import { Test, console2, safeconsole as console } from "forge-std/Test.sol";

interface IOpenVmHalo2Verifier {
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit) external view;

    function LEAF_EXE_COMMIT() external view returns (bytes32);
}

contract TemplateTest is Test {
    bytes partialProof;
    bytes32 appExeCommit = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
    bytes guestPvs;
    IOpenVmHalo2Verifier verifier = IOpenVmHalo2Verifier(address(bytes20(keccak256(abi.encode("OpenVmHalo2Verifier")))));

    uint256 publicValuesLength;
    uint256 fullProofWords;
    uint256 fullProofLength;

    function setUp() public {
        partialProof = new bytes(55 * 32);
        for (uint256 i = 0; i < 55; i++) {
            for (uint256 j = 0; j < 32; j++) {
                partialProof[i * 32 + j] = bytes1(uint8(i));
            }
        }
    }

    /// forge-config: default.fuzz.runs = 100
    function testFuzz_ProofFormat(uint256 _publicValuesLength) public {
        publicValuesLength = bound(_publicValuesLength, 1, 5000);
        fullProofWords = (12 + 2 + publicValuesLength + 43);
        fullProofLength = fullProofWords * 32;

        guestPvs = new bytes(publicValuesLength);
        for (uint256 i = 0; i < publicValuesLength; i++) {
            guestPvs[i] = bytes1(uint8(i));
        }

        bytes memory compiledVerifier = _compileOpenVmVerifier(publicValuesLength);

        console2.logBytes(compiledVerifier);

        assembly {
            let addr := create(0, add(compiledVerifier, 0x20), mload(compiledVerifier))
            if iszero(extcodesize(addr)) { revert(0, 0) }
            sstore(verifier.slot, addr)
        }

        (bool success,) = address(verifier).delegatecall(
            abi.encodeWithSelector(IOpenVmHalo2Verifier.verify.selector, guestPvs, partialProof, appExeCommit)
        );
        require(success, "Verification failed");
    }

    fallback(bytes calldata proof) external returns (bytes memory) {
        bytes memory partialProofExpected = partialProof;

        uint256 guestPvsSuffixOffset = 0x1c0 + (32 * publicValuesLength);

        bytes memory kzgAccumulators = proof[0:0x180];
        bytes memory guestPvsSuffix = proof[guestPvsSuffixOffset:];
        bytes memory _partialProof = abi.encodePacked(kzgAccumulators, guestPvsSuffix);

        require(keccak256(_partialProof) == keccak256(partialProofExpected), "Partial proof mismatch");

        bytes memory _appExeCommit = proof[0x180:0x1a0];
        bytes memory _leafExeCommit = proof[0x1a0:0x1c0];

        require(bytes32(_appExeCommit) == appExeCommit, "App exe commit mismatch");
        require(bytes32(_leafExeCommit) == verifier.LEAF_EXE_COMMIT(), "Leaf exe commit mismatch");

        bytes calldata _guestPvs = proof[0x1c0:0x1c0 + 32 * publicValuesLength];
        for (uint256 i = 0; i < publicValuesLength; ++i) {
            uint256 expected = uint256(uint8(guestPvs[i]));
            uint256 actual = uint256(bytes32(_guestPvs[i * 32:(i + 1) * 32]));
            require(expected == actual, "Guest PVs hash mismatch");
        }
    }

    function _compileOpenVmVerifier(uint256 _publicValuesLength) private returns (bytes memory) {
        string memory const = string.concat(
            "\nuint256 constant PUBLIC_VALUES_LENGTH = ", LibString.toString(_publicValuesLength), ";\n\n"
        );

        // `code` will look like this:
        //
        // // SPDX-License-Identifier: MIT
        // pragma solidity 0.8.19;
        //
        // import { Halo2Verifier } ...
        // import { IOpenVmHalo2Verifier } ...
        //
        // contract OpenVmHalo2Verifier { .. }
        string memory code = vm.readFile("template/OpenVmHalo2Verifier.sol");

        string memory deps = vm.readFile("test/helpers/ExampleDeps.sol");

        // We want to replace the `import` statements with inlined deps for JIT
        // compilation.
        string memory inlinedCode = LibString.replace(
            code,
            "import { Halo2Verifier } from \"./Halo2Verifier.sol\";\nimport { IOpenVmHalo2Verifier } from \"./interfaces/IOpenVmHalo2Verifier.sol\";",
            deps
        );

        // Must use solc 0.8.19
        string[] memory commands = new string[](3);
        commands[0] = "sh";
        commands[1] = "-c";
        commands[2] = string.concat(
            "echo ",
            "'",
            const,
            inlinedCode,
            "'",
            " | solc --no-optimize-yul --bin --optimize --optimize-runs 100000 - ",
            " | awk 'BEGIN{found=0} /:OpenVmHalo2Verifier/ {found=1; next} found && /^Binary:/ {getline; print; exit}'"
        );

        console2.log(commands[2]);

        return vm.ffi(commands);
    }
}
