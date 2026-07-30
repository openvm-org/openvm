| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 456 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 7,245 |  14,365,133 |  1,532 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 4,710 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 655 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 304 |  592,827 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-9892fb04775004ed5d774fe5b65ff1921a74c6f9.md) | 2,659 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9892fb04775004ed5d774fe5b65ff1921a74c6f9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30551746908)
