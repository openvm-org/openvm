| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/fibonacci-b102763dad89f20a1e8726498ca3135746a50418.md) | 469 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/keccak-b102763dad89f20a1e8726498ca3135746a50418.md) | 7,348 |  14,365,133 |  1,538 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/sha2_bench-b102763dad89f20a1e8726498ca3135746a50418.md) | 4,696 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/regex-b102763dad89f20a1e8726498ca3135746a50418.md) | 674 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/ecrecover-b102763dad89f20a1e8726498ca3135746a50418.md) | 228 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/pairing-b102763dad89f20a1e8726498ca3135746a50418.md) | 321 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3051/kitchen_sink-b102763dad89f20a1e8726498ca3135746a50418.md) | 2,662 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b102763dad89f20a1e8726498ca3135746a50418

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30123881545)
