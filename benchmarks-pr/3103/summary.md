| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 462 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 7,399 |  14,365,133 |  1,508 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 4,175 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 649 |  4,090,656 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 196 |  112,210 |  199 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 235 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3.md) | 2,019 |  1,979,971 |  520 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6b08efdaa65df44fdbe5bb3122b42a88bd49bfb3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32175899510)
