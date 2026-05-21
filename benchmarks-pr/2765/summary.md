| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 1,866 |  4,000,051 |  514 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 13,624 |  14,365,133 |  2,232 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 9,524 |  11,167,961 |  1,411 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 1,558 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 604 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 742 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-95590ce9fff7402984f1441191a5ca5c9eefa1ce.md) | 1,878 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/95590ce9fff7402984f1441191a5ca5c9eefa1ce

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26249690222)
