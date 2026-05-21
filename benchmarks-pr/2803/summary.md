| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/fibonacci-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 3,718 |  12,000,265 |  909 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/keccak-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 18,518 |  18,655,329 |  3,293 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/sha2_bench-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 10,194 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/regex-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 1,376 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/ecrecover-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 596 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/pairing-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 885 |  1,745,757 |  270 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2803/kitchen_sink-2759e004f8e6b10da8b21968ca13d0996fa74c6e.md) | 1,906 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2759e004f8e6b10da8b21968ca13d0996fa74c6e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26244590895)
