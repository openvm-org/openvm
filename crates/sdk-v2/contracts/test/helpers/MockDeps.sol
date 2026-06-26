interface IOpenVmHalo2Verifier {
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit, bytes32 appVmCommit)
        external
        view;
}

contract Halo2Verifier {
    /// Mock verifier always reverts
    fallback(bytes calldata) external returns (bytes memory) {
        revert("Verification failed");
    }
}
