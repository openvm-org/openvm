| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/fibonacci-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 3,764 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/keccak-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 18,721 |  18,655,329 |  3,298 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/sha2_bench-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 10,071 |  14,793,960 |  1,447 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/regex-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 1,422 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/ecrecover-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 613 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/pairing-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 892 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2833/kitchen_sink-c11bec887f241c6f75ce24166240b0ebc072a16d.md) | 1,907 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c11bec887f241c6f75ce24166240b0ebc072a16d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26776347290)
