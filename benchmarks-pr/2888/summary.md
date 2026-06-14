| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/fibonacci-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 3,040 |  12,000,265 |  666 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/keccak-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 16,351 |  18,655,329 |  3,049 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/sha2_bench-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 9,249 |  14,793,960 |  1,131 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/regex-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 1,154 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/ecrecover-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 607 |  123,583 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/pairing-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 935 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2888/kitchen_sink-0afb9a81f58cf272cde2e1bc5beb200471cdeb0e.md) | 4,090 |  2,579,903 |  876 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0afb9a81f58cf272cde2e1bc5beb200471cdeb0e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27494920737)
