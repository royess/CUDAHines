# README

The project contains serial and parallel, i.e. CUDA, implementations of Hines algorithm.

## Build

Build by `script/Makefile`. Build serial and parallel versions:

```bash
make
```

Build serial version:

```bash
make serial
```

Build serial and parallel versions:

```bash
make parallel
```

The executables are built into `bin`.

## Run and Test

### Single Run

Run the serial version on case 1:

```bash
./serial ../data/case1.txt ../sresult/res1.txt
```

Run the verification executable on case 1:

```bash
./check ../data/case1.txt ../cresult/res1.txt
```

### Multiple Runs

You can also run a single case for multiple times.

Run the serial version on case 1 for 256 times:

```bash
./serial ../data/case1.txt ../sresult/res1.txt 256
```

Run the parallel version on case 1 for 256 times:

```bash
./parallel ../data/case1.txt ../presult/res1.txt 256
```

**Note that the number of runs should be a multiple of 32. This requirement is for GPU to well perform.**

### Python Script

The script `script/measure_time.py` is for testing purpose. (It requires Python version >= 3.6.) You can modify the variable `run_dict`to test varied test cases and the number of runs.

## Test Result

Test platform: Tesla K80. The result is stored in `script/time_cost.txt`.
