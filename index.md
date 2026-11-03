| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 1,718 |  12,000,265 |  375 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 9,648 |  18,655,329 |  1,546 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 5,317 |  14,793,960 |  586 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 692 |  4,137,067 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 439 |  123,583 |  193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 561 |  1,745,757 |  200 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-b4f4f7c26a0fe4215dfe2af01662d3fab1905c56.md) | 2,302 |  2,579,903 |  493 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b4f4f7c26a0fe4215dfe2af01662d3fab1905c56

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/41680522128)
