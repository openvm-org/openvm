| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 412 |  4,000,051 |  237 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 8,616 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 4,190 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 573 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 218 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 283 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-448f3dfc23beabe8c2c405f233f21ba468d9bd01.md) | 1,938 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/448f3dfc23beabe8c2c405f233f21ba468d9bd01

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29655582504)
