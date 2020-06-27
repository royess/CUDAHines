# README

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

# Test

Run the serial version on case 1:

```bash
./serial ../data/case1.txt ../sresult/res1.txt
```

Run the parallel version on case 1:

```bash
./parallel ../data/case1.txt ../presult/res1.txt
```

Run the verification executable on case 1:
```bash
./check ../data/case1.txt ../cresult/res1.txt
```
