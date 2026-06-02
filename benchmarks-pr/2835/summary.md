| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/fibonacci-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 3,788 |  12,000,265 |  924 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/keccak-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 18,151 |  18,655,329 |  3,299 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/sha2_bench-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 9,906 |  14,793,960 |  1,441 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/regex-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 1,423 |  4,137,067 |  362 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/ecrecover-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 606 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/pairing-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 885 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2835/kitchen_sink-65934b13d030a6580c552b98a4bac7ea512106c4.md) | 1,864 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/65934b13d030a6580c552b98a4bac7ea512106c4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26827860154)
