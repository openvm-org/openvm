| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 1,841 |  4,000,051 |  456 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 14,021 |  14,365,133 |  2,228 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 8,279 |  11,167,961 |  915 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 1,599 |  4,090,656 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 641 |  112,210 |  293 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 755 |  592,827 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-ef88ae679a266d8dbd673584357a860625cd2f36.md) | 2,013 |  1,979,971 |  430 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ef88ae679a266d8dbd673584357a860625cd2f36

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25881949932)
