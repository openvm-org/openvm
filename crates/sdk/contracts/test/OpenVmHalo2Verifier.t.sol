// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { LibString } from "./helpers/LibString.sol";
import { Test, console2, safeconsole as console } from "forge-std/Test.sol";
import { IOpenVmHalo2Verifier } from "../src/IOpenVmHalo2Verifier.sol";

contract TemplateTest is Test {
    bytes proofData;
    bytes32 appExeCommit = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
    bytes32 appVmCommit = 0xEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE;
    bytes guestPvs;

    uint256 publicValuesLength;
    uint256 fullProofWords;
    uint256 fullProofLength;

    string _code = vm.readFile("template/OpenVmHalo2Verifier.sol");
    string deps = vm.readFile("test/helpers/MockDeps.sol");

    function setUp() public {
        proofData = new bytes(55 * 32);
        for (uint256 i = 0; i < 55; i++) {
            for (uint256 j = 0; j < 32; j++) {
                proofData[i * 32 + j] = bytes1(uint8(i));
            }
        }
    }

    /// forge-config: default.fuzz.runs = 10
    function testFuzz_ProofFormat(uint256 _publicValuesLength) public {
        publicValuesLength = bound(_publicValuesLength, 1, 10_000);
        publicValuesLength = 8;
        fullProofWords = (12 + 2 + publicValuesLength + 43);
        fullProofLength = fullProofWords * 32;

        guestPvs = new bytes(publicValuesLength);
        for (uint256 i = 0; i < publicValuesLength; i++) {
            guestPvs[i] = bytes1(uint8(i));
        }

        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        (bool success,) = address(verifier).delegatecall(
            abi.encodeCall(IOpenVmHalo2Verifier.verify, (guestPvs, proofData, appExeCommit, appVmCommit))
        );
        require(success, "Verification failed");
    }

    fallback(bytes calldata proof) external returns (bytes memory) {
        bytes memory proofDataExpected = proofData;

        uint256 proofSuffixOffset = 0x1c0 + (32 * publicValuesLength);

        bytes memory kzgAccumulators = proof[0:0x180];
        bytes memory proofSuffix = proof[proofSuffixOffset:];
        bytes memory _proofData = abi.encodePacked(kzgAccumulators, proofSuffix);

        require(keccak256(_proofData) == keccak256(proofDataExpected), "Partial proof mismatch");

        bytes memory _appExeCommit = proof[0x180:0x1a0];
        bytes memory _appVmCommit = proof[0x1a0:0x1c0];

        require(bytes32(_appExeCommit) == appExeCommit, "App exe commit mismatch");
        require(bytes32(_appVmCommit) == appVmCommit, "App vm commit mismatch");

        bytes calldata _guestPvs = proof[0x1c0:0x1c0 + 32 * publicValuesLength];
        for (uint256 i = 0; i < publicValuesLength; ++i) {
            uint256 expected = uint256(uint8(guestPvs[i]));
            uint256 actual = uint256(bytes32(_guestPvs[i * 32:(i + 1) * 32]));
            require(expected == actual, "Guest PVs hash mismatch");
        }

        // Suppress return value warning
        assembly {
            return(0x00, 0x00)
        }
    }

    function test_RevertWhen_InvalidPublicValuesLength() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory invalidPvs = new bytes(0);
        bytes4 sig = bytes4(keccak256("InvalidPublicValuesLength(uint256,uint256)"));

        vm.expectRevert(abi.encodeWithSelector(sig, 32, invalidPvs.length));
        verifier.verify(invalidPvs, hex"", bytes32(0), bytes32(0));
    }

    function test_RevertWhen_InvalidProofDataLength() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory invalidProofData = new bytes(0);
        bytes4 sig = bytes4(keccak256("InvalidProofDataLength(uint256,uint256)"));

        bytes memory pvs = new bytes(publicValuesLength);

        vm.expectRevert(abi.encodeWithSelector(sig, 55 * 32, invalidProofData.length));
        verifier.verify(pvs, invalidProofData, appExeCommit, appVmCommit);
    }

    function test_RevertWhen_ProofVerificationFailed() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory _proofData = new bytes(55 * 32);
        bytes memory pvs = new bytes(publicValuesLength);

        bytes4 sig = bytes4(keccak256("ProofVerificationFailed()"));

        vm.expectRevert(abi.encodeWithSelector(sig));
        verifier.verify(pvs, _proofData, appExeCommit, appVmCommit);
    }

    function _compileAndDeployOpenVmVerifier(uint256 _publicValuesLength)
        private
        returns (IOpenVmHalo2Verifier verifier)
    {
        string memory code = LibString.replace(_code, "{PUBLIC_VALUES_LENGTH}", LibString.toString(_publicValuesLength));

        // `code` will look like this:
        //
        // // SPDX-License-Identifier: MIT
        // pragma solidity 0.8.19;
        //
        // import { Halo2Verifier } ...
        // import { IOpenVmHalo2Verifier } ...
        //
        // contract OpenVmHalo2Verifier { .. }
        //
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
            inlinedCode,
            "'",
            " | solc --no-optimize-yul --bin --optimize --optimize-runs 100000 - ",
            " | awk 'BEGIN{found=0} /:OpenVmHalo2Verifier/ {found=1; next} found && /^Binary:/ {getline; print; exit}'"
        );

        bytes memory compiledVerifier = vm.ffi(commands);

        assembly {
            verifier := create(0, add(compiledVerifier, 0x20), mload(compiledVerifier))
            if iszero(extcodesize(verifier)) { revert(0, 0) }
        }
    }
}
