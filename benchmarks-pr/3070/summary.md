| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/fibonacci-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 477 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/keccak-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 7,640 |  14,365,133 |  1,626 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/sha2_bench-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 4,412 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/regex-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 779 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/ecrecover-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 209 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/pairing-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 248 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3070/kitchen_sink-6a4a53a1bd53c9e9704bc29f65ee555d30eba27c.md) | 2,225 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6a4a53a1bd53c9e9704bc29f65ee555d30eba27c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34331527508)
