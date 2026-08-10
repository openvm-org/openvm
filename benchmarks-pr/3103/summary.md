| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 467 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 7,500 |  14,365,133 |  1,551 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 4,140 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 652 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 222 |  112,210 |  194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 233 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-ed49dd99505599e2bb41d3aab2e0437aa774b173.md) | 2,033 |  1,979,971 |  527 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ed49dd99505599e2bb41d3aab2e0437aa774b173

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31416086596)
