| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 448 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 7,211 |  14,365,133 |  1,594 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 4,022 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 713 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 207 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 237 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-db8791d7648bdea34dd807d1afc2c1807b91dc64.md) | 2,143 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db8791d7648bdea34dd807d1afc2c1807b91dc64

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32430905991)
