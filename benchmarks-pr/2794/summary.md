| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/fibonacci-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 1,608 |  4,000,051 |  451 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/keccak-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 14,005 |  14,365,133 |  2,396 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/sha2_bench-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 9,229 |  11,167,961 |  1,403 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/regex-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 1,490 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/ecrecover-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 479 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/pairing-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 589 |  592,827 |  253 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2794/kitchen_sink-075b593d1f98128c723733d45ff0acf38a44bb66.md) | 1,827 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/075b593d1f98128c723733d45ff0acf38a44bb66

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26486554450)
