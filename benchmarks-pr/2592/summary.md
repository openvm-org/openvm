| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 3,818 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 18,698 |  18,655,329 |  3,337 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 1,418 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 655 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 899 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-51b2eedbaf826cde999f5856b7aa0c538dd74030.md) | 2,275 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/51b2eedbaf826cde999f5856b7aa0c538dd74030

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23924853926)
