| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 1,033 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 15,976 |  14,365,133 |  3,075 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 8,164 |  11,167,961 |  1,004 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 1,182 |  4,090,656 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 435 |  112,210 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 589 |  592,827 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-33233523e9ad53a2255c8b43c0714f8003a3dce1.md) | 3,874 |  1,979,971 |  870 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/33233523e9ad53a2255c8b43c0714f8003a3dce1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28323635989)
