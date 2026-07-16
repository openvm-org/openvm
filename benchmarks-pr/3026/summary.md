| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/fibonacci-4880e826e1b1393da4da366f28330eb172b39a56.md) | 415 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/keccak-4880e826e1b1393da4da366f28330eb172b39a56.md) | 8,419 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/sha2_bench-4880e826e1b1393da4da366f28330eb172b39a56.md) | 3,970 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/regex-4880e826e1b1393da4da366f28330eb172b39a56.md) | 568 |  4,090,656 |  209 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/ecrecover-4880e826e1b1393da4da366f28330eb172b39a56.md) | 219 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/pairing-4880e826e1b1393da4da366f28330eb172b39a56.md) | 278 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3026/kitchen_sink-4880e826e1b1393da4da366f28330eb172b39a56.md) | 1,887 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4880e826e1b1393da4da366f28330eb172b39a56

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29506822946)
