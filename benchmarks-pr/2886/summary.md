| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/fibonacci-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 3,924 |  12,000,265 |  1,142 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/keccak-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 21,715 |  18,655,329 |  4,598 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/sha2_bench-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 9,663 |  14,793,960 |  1,862 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/regex-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 1,520 |  4,137,067 |  433 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/ecrecover-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 600 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/pairing-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 933 |  1,745,757 |  304 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2886/kitchen_sink-5770005c7ee7e3a4b25d6af7800dd40e0d87205f.md) | 4,132 |  2,579,903 |  886 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5770005c7ee7e3a4b25d6af7800dd40e0d87205f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27436389723)
