| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/fibonacci-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 3,800 |  12,000,265 |  954 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/keccak-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 18,591 |  18,655,329 |  3,312 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/regex-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 1,448 |  4,137,067 |  384 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/ecrecover-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 645 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/pairing-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 902 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2692/kitchen_sink-1241cde3760496d68e62429ccba6c271fde0ae08.md) | 2,153 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1241cde3760496d68e62429ccba6c271fde0ae08

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24257000569)
