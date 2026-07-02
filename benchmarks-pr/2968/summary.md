| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/fibonacci-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 3,040 |  12,000,265 |  673 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/keccak-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 16,416 |  18,655,329 |  3,068 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/sha2_bench-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 9,216 |  14,793,960 |  1,129 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/regex-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 1,169 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/ecrecover-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 608 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/pairing-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 944 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2968/kitchen_sink-8fb30decac46d12a2ecb85e331e7ca2dc84d433a.md) | 4,089 |  2,579,903 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8fb30decac46d12a2ecb85e331e7ca2dc84d433a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28620336081)
