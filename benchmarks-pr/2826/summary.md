| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/fibonacci-35463c95793fb7def522b623166e8072c751033c.md) | 3,753 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/keccak-35463c95793fb7def522b623166e8072c751033c.md) | 18,386 |  18,655,329 |  3,240 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/sha2_bench-35463c95793fb7def522b623166e8072c751033c.md) | 10,173 |  14,793,960 |  1,445 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/regex-35463c95793fb7def522b623166e8072c751033c.md) | 1,403 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/ecrecover-35463c95793fb7def522b623166e8072c751033c.md) | 606 |  123,583 |  255 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/pairing-35463c95793fb7def522b623166e8072c751033c.md) | 887 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2826/kitchen_sink-35463c95793fb7def522b623166e8072c751033c.md) | 1,892 |  2,579,903 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/35463c95793fb7def522b623166e8072c751033c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26907601436)
