// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

import { LibString } from "./helpers/LibString.sol";
import { Test, console2, safeconsole as console, stdError } from "forge-std/Test.sol";
import { IOpenVmHalo2Verifier } from "../src/IOpenVmHalo2Verifier.sol";

contract TemplateTest is Test {
    // BN254 scalar field modulus (Fr), as specified in EIP-197:
    // https://eips.ethereum.org/EIPS/eip-197
    uint256 constant BN254_SCALAR_MODULUS = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001;

    bytes proofData;
    bytes32 appExeCommit = 0x2222222222222222222222222222222222222222222222222222222222222222;
    bytes32 appVmCommit = 0x1111111111111111111111111111111111111111111111111111111111111111;
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

        (bool success,) = address(verifier)
            .delegatecall(abi.encodeCall(IOpenVmHalo2Verifier.verify, (guestPvs, proofData, appExeCommit, appVmCommit)));
        require(success, "Verification failed");
    }

    fallback(bytes calldata proof) external returns (bytes memory) {
        bytes memory proofDataExpected = proofData;

        uint256 proofSuffixOffset = 0x1c0 + (32 * publicValuesLength);

        bytes memory kzgAccumulator = proof[0:0x180];
        bytes memory proofSuffix = proof[proofSuffixOffset:];
        bytes memory _proofData = abi.encodePacked(kzgAccumulator, proofSuffix);

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

    function test_RevertWhen_InvalidAppExeCommit() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory pvs = new bytes(publicValuesLength);
        bytes memory _proofData = new bytes(55 * 32);
        bytes32 invalidAppExeCommit = bytes32(BN254_SCALAR_MODULUS);
        bytes4 sig = bytes4(keccak256("InvalidAppExeCommit(bytes32)"));

        vm.expectRevert(abi.encodeWithSelector(sig, invalidAppExeCommit));
        verifier.verify(pvs, _proofData, invalidAppExeCommit, appVmCommit);
    }

    function test_RevertWhen_InvalidAppVmCommit() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory pvs = new bytes(publicValuesLength);
        bytes memory _proofData = new bytes(55 * 32);
        bytes32 invalidAppVmCommit = bytes32(BN254_SCALAR_MODULUS);
        bytes4 sig = bytes4(keccak256("InvalidAppVmCommit(bytes32)"));

        vm.expectRevert(abi.encodeWithSelector(sig, invalidAppVmCommit));
        verifier.verify(pvs, _proofData, appExeCommit, invalidAppVmCommit);
    }

    function test_OnlyVerifySelectorIsExposed() public {
        bytes memory methodIdentifiers = _compiledOpenVmVerifierMethodIdentifiers(32);
        assertEq(string(methodIdentifiers), "24270d54: verify(bytes,bytes,bytes32,bytes32)");
    }

    function test_RevertWhen_ProofDataPrefixIsNonZero() public {
        publicValuesLength = 32;
        IOpenVmHalo2Verifier verifier = _compileAndDeployOpenVmVerifier(publicValuesLength);

        bytes memory pvs = new bytes(publicValuesLength);
        bytes memory invalidProofData = proofData;
        invalidProofData[0] = bytes1(uint8(1));

        vm.expectRevert(stdError.assertionError);
        verifier.verify(pvs, invalidProofData, appExeCommit, appVmCommit);
    }

    function test_Bn254ScalarModulusMatchesEcmulPrecompile() public view {
        (uint256 qx, uint256 qy) = _ecMul(1, 2, BN254_SCALAR_MODULUS);
        assertEq(qx, 0, "q * G should be point at infinity");
        assertEq(qy, 0, "q * G should be point at infinity");

        (uint256 qPlusOneX, uint256 qPlusOneY) = _ecMul(1, 2, BN254_SCALAR_MODULUS + 1);
        assertEq(qPlusOneX, 1, "(q + 1) * G should wrap to G");
        assertEq(qPlusOneY, 2, "(q + 1) * G should wrap to G");
    }

    function _compileAndDeployOpenVmVerifier(uint256 _publicValuesLength)
        private
        returns (IOpenVmHalo2Verifier verifier)
    {
        string memory inlinedCode = _inlinedOpenVmVerifierCode(_publicValuesLength);

        // Must use solc 0.8.19
        string[] memory commands = new string[](3);
        commands[0] = "sh";
        commands[1] = "-c";
        commands[2] = string.concat(
            "cat <<'SOL' | solc --no-optimize-yul --bin --optimize --optimize-runs 100000 - ",
            " | awk 'BEGIN{found=0} /:OpenVmHalo2Verifier/ {found=1; next} found && /^Binary:/ {getline; print; exit}'\n",
            inlinedCode,
            "\nSOL\n"
        );

        bytes memory compiledVerifier = vm.ffi(commands);

        assembly {
            verifier := create(0, add(compiledVerifier, 0x20), mload(compiledVerifier))
            if iszero(extcodesize(verifier)) { revert(0, 0) }
        }
    }

    function _compiledOpenVmVerifierMethodIdentifiers(uint256 _publicValuesLength) private returns (bytes memory) {
        string memory inlinedCode = _inlinedOpenVmVerifierCode(_publicValuesLength);

        string[] memory commands = new string[](3);
        commands[0] = "sh";
        commands[1] = "-c";
        commands[2] = string.concat(
            "cat <<'SOL' | solc --combined-json hashes - | jq -r '.contracts[\"<stdin>:OpenVmHalo2Verifier\"].hashes | to_entries | sort_by(.key) | map(\"\\(.value): \\(.key)\") | join(\"\\n\")'\n",
            inlinedCode,
            "\nSOL\n"
        );

        return vm.ffi(commands);
    }

    function _inlinedOpenVmVerifierCode(uint256 _publicValuesLength) private view returns (string memory) {
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
        return LibString.replace(
            code,
            "import { Halo2Verifier } from \"./Halo2Verifier.sol\";\nimport { IOpenVmHalo2Verifier } from \"./interfaces/IOpenVmHalo2Verifier.sol\";",
            deps
        );
    }

    function _ecMul(uint256 x, uint256 y, uint256 scalar) private view returns (uint256 rx, uint256 ry) {
        (bool success, bytes memory result) = address(0x07).staticcall(abi.encode(x, y, scalar));
        require(success, "ecmul precompile failed");
        (rx, ry) = abi.decode(result, (uint256, uint256));
    }
}
