| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/fibonacci-846f02af33a476c73245d72d591b7bd562803a0b.md) | 1,036 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/keccak-846f02af33a476c73245d72d591b7bd562803a0b.md) | 15,601 |  14,365,133 |  2,994 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/sha2_bench-846f02af33a476c73245d72d591b7bd562803a0b.md) | 8,134 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/regex-846f02af33a476c73245d72d591b7bd562803a0b.md) | 1,179 |  4,090,656 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/ecrecover-846f02af33a476c73245d72d591b7bd562803a0b.md) | 435 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/pairing-846f02af33a476c73245d72d591b7bd562803a0b.md) | 600 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2939/kitchen_sink-846f02af33a476c73245d72d591b7bd562803a0b.md) | 3,862 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/846f02af33a476c73245d72d591b7bd562803a0b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28391249906)
