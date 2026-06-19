| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/fibonacci-d273a0629899a36832229b4b968a0998fb531590.md) | 3,001 |  12,000,265 |  660 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/keccak-d273a0629899a36832229b4b968a0998fb531590.md) | 16,179 |  18,655,329 |  3,004 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/sha2_bench-d273a0629899a36832229b4b968a0998fb531590.md) | 9,166 |  14,793,960 |  1,131 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/regex-d273a0629899a36832229b4b968a0998fb531590.md) | 1,169 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/ecrecover-d273a0629899a36832229b4b968a0998fb531590.md) | 600 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/pairing-d273a0629899a36832229b4b968a0998fb531590.md) | 935 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/kitchen_sink-d273a0629899a36832229b4b968a0998fb531590.md) | 4,093 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d273a0629899a36832229b4b968a0998fb531590

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27832366976)
