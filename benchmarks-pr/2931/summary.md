| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/fibonacci-e3c511e7e53de18991d55054091255053799813a.md) | 1,037 |  4,000,051 |  394 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/keccak-e3c511e7e53de18991d55054091255053799813a.md) | 15,641 |  14,365,133 |  3,010 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/sha2_bench-e3c511e7e53de18991d55054091255053799813a.md) | 8,215 |  11,167,961 |  1,009 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/regex-e3c511e7e53de18991d55054091255053799813a.md) | 1,177 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/ecrecover-e3c511e7e53de18991d55054091255053799813a.md) | 457 |  112,210 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/pairing-e3c511e7e53de18991d55054091255053799813a.md) | 583 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2931/kitchen_sink-e3c511e7e53de18991d55054091255053799813a.md) | 3,875 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e3c511e7e53de18991d55054091255053799813a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28321762472)
