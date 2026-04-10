| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-895a74c8d16713bdf25db11fee065ea02475863a.md) | 3,853 |  12,000,265 |  957 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-895a74c8d16713bdf25db11fee065ea02475863a.md) | 18,440 |  18,655,329 |  3,313 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-895a74c8d16713bdf25db11fee065ea02475863a.md) | 1,448 |  4,137,067 |  381 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-895a74c8d16713bdf25db11fee065ea02475863a.md) | 645 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-895a74c8d16713bdf25db11fee065ea02475863a.md) | 899 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-895a74c8d16713bdf25db11fee065ea02475863a.md) | 2,171 |  2,579,903 |  439 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/895a74c8d16713bdf25db11fee065ea02475863a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24257675982)
