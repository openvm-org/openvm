# Testing Framework

The testing framework provides scaffolding to test a single chip at a time.
To do this, a test harness provide dummy chips that add unconstrained messages to buses to balance the buses. The primary chips necessary are:

- `ExecutionTester` to add instructions to EXECUTION_BUS
- `MemoryTester` to add memory writes to initialize memory with test input data. `MemoryTester` can also be used to read memory to check for expected results.
