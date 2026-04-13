| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/fibonacci-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 3,946 |  12,000,265 |  974 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/keccak-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 18,598 |  18,655,329 |  3,305 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/sha2_bench-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 8,959 |  14,793,960 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/regex-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 1,423 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/ecrecover-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 646 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/pairing-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 909 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2695/kitchen_sink-9a0f2373e0504cdfd28336f4d2a772a535d614e4.md) | 2,089 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9a0f2373e0504cdfd28336f4d2a772a535d614e4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24361308248)
