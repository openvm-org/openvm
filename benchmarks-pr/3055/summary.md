| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/fibonacci-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 479 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/keccak-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 7,349 |  14,365,133 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/sha2_bench-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 4,766 |  11,167,961 |  536 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/regex-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 676 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/ecrecover-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 229 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/pairing-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 326 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3055/kitchen_sink-893c6c372a6f2f96b5aa5667b817bb9bf36044e8.md) | 2,689 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/893c6c372a6f2f96b5aa5667b817bb9bf36044e8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29944212188)
