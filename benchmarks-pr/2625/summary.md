| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 3,784 |  12,000,265 |  934 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 15,640 |  1,235,218 |  2,164 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 1,415 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 661 |  122,348 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 918 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-ea2703bb0883b0ce99ac7f23e2aecf929d5fff68.md) | 2,374 |  154,763 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ea2703bb0883b0ce99ac7f23e2aecf929d5fff68

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23759743158)
