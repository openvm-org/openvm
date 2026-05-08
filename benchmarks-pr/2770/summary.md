| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/fibonacci-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 3,886 |  12,000,265 |  965 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/keccak-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 18,590 |  18,655,329 |  3,309 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/sha2_bench-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 8,943 |  14,793,960 |  1,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/regex-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 1,417 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/ecrecover-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 649 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/pairing-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 901 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2770/kitchen_sink-acb043a87647e6f24da3ea578e74f583c68fe269.md) | 2,080 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/acb043a87647e6f24da3ea578e74f583c68fe269

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25575814988)
