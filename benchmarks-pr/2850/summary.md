| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 5,240 |  4,000,051 |  434 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 18,585 |  14,365,133 |  2,361 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 12,478 |  11,167,961 |  1,408 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 3,692 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 1,954 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 2,067 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-90b629cd19b4d9f996e3328f05ca39674c9ef942.md) | 5,973 |  1,979,971 |  937 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/90b629cd19b4d9f996e3328f05ca39674c9ef942

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27307268173)
