| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 1,677 |  12,000,265 |  370 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 9,535 |  18,655,329 |  1,543 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 5,300 |  14,793,960 |  592 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 703 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 430 |  123,583 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 586 |  1,745,757 |  194 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-643850e48946d7de8c3e28fd1a9bb78dc2b279ae.md) | 2,298 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/643850e48946d7de8c3e28fd1a9bb78dc2b279ae

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33403397967)
