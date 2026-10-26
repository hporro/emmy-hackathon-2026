### V0.0.1
- Asked claude to make a first version of the compiler for phase 1. It followed TASK.md so it has 4 compilation phases: load, prologue, walk and emit.
- It passes the 4 tests for the phase 1.
- Seems to have a good structure, altough not as simple as possible.
- claude also made the run_all.py script in order to run the tests easily.

### V0.0.2
- Asked claude to complete the other phases, while keeping the code as simple as possible. Now it only makes a emmiter class with a static prologue.
- It passes all tests.

### v1.0.0
- Now all the tests run and I asked claude to make a little refactor of the code.
- Added more operations to the F32_OPS list, in order to support more of them (closing the gap between what the IR can express and what the compiler can do).
- Added typed expressions, in an effort into making the code readable.

### V1.0.1
- Big refactor: Expressions and statements are classes now.

### V1.0.2
- Added some extra problems to test the compiler on, and added an operation to make some tests pass. Updated the run_all.py script in order to run the extra tests.
- Figured out that in order to make the tests that are not passing pass, we have to support f16 types.
- Figured out that integer types are not correctly handled.