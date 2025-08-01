# Solidity SDK

As a supplement to OpenVM, we provide a [Solidity SDK](https://github.com/openvm-org/openvm-solidity-sdk) containing OpenVM verifier contracts generated at official release commits using the `cargo openvm setup` [command](../advanced-usage/sdk.md#setup). The contracts are built at every _minor_ release as OpenVM guarantees verifier backward compatibility across patch releases.

Note that these builds are for the default aggregation VM config which should be sufficient for most users. If you use a custom config, you will need to manually generate the verifier contract using the [OpenVM SDK](../advanced-usage/sdk.md).

## Installation

To install the Solidity SDK as a dependency into your forge project, run the following command:

```bash
forge install openvm-org/openvm-solidity-sdk
```

## Usage

Once you have the SDK installed, you can import the SDK contracts into your Solidity project:

```solidity
import "openvm-solidity-sdk/v1.3/OpenVmHalo2Verifier.sol";
```

If you are using an already-deployed verifier contract, you can simply import the `IOpenVmHalo2Verifier` interface:

```solidity
import { IOpenVmHalo2Verifier } from "openvm-solidity-sdk/v1.3/interfaces/IOpenVmHalo2Verifier.sol";

contract MyContract {
    function myFunction() public view {
        // ... snip ...

        IOpenVmHalo2Verifier(verifierAddress)
            .verify(publicValues, proofData, appExeCommit, appVmCommit);

        // ... snip ...
    }
}
```

The arguments to the `verify` function are the fields in the [EVM Proof JSON Format](./verify.md#evm-proof-json-format).
Since the builds use the default aggregation VM config, the number of public values is fixed to 32.

If you want to import the verifier contract into your own repository for testing purposes, note that it is locked to Solidity version `0.8.19`. If your project uses a different version, the import may not compile. As a workaround, you can compile the contract separately and use `vm.etch()` to inject the raw bytecode into your tests.

## Deployment

To deploy an instance of a verifier contract, you can clone the repo and simply use `forge create`:

```bash
git clone --recursive https://github.com/openvm-org/openvm-solidity-sdk.git
cd openvm-solidity-sdk
forge create src/v1.3/OpenVmHalo2Verifier.sol:OpenVmHalo2Verifier --rpc-url $RPC --private-key $PRIVATE_KEY --broadcast
```

We recommend a direct deployment from the SDK repo since the proper compiler configurations are all pre-set.
