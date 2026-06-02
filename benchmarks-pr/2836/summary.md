| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/fibonacci-6212fc57df378d98f78ec53368597507d32bb115.md) | 3,749 |  12,000,265 |  927 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/keccak-6212fc57df378d98f78ec53368597507d32bb115.md) | 18,052 |  18,655,329 |  3,271 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/sha2_bench-6212fc57df378d98f78ec53368597507d32bb115.md) | 10,046 |  14,793,960 |  1,460 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/regex-6212fc57df378d98f78ec53368597507d32bb115.md) | 1,416 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/ecrecover-6212fc57df378d98f78ec53368597507d32bb115.md) | 603 |  123,583 |  258 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/pairing-6212fc57df378d98f78ec53368597507d32bb115.md) | 881 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2836/kitchen_sink-6212fc57df378d98f78ec53368597507d32bb115.md) | 3,859 |  2,579,903 |  951 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6212fc57df378d98f78ec53368597507d32bb115

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26828156946)
