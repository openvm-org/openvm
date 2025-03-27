interface IOpenVmHalo2Verifier {
    function verify(bytes calldata publicValues, bytes calldata proofData, bytes32 appExeCommit) external view;
}

contract Halo2Verifier {
    fallback(bytes calldata) external returns (bytes memory) { }
}
