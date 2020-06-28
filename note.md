# Note

**Deadline: June. 28th**

To-do list:

- [x] Complete serial code. Verify it.
- [x] Complete CUDA code.  

## Details

- Hotspot 是读写.
- 为了可以对同一个 case 进行多次测试, 把读取的数组拷贝到另一个数组, 再进行计算. 这会引入内存读写的额外时间, 但是运行次数较多的情况下, 额外运行时间不大, 可以不太考虑.
- 两种情况下表现很差:
    - 数据太大. case 11, 12. 因为内存限制, 不能开很多的 thread.
    - 数据太小. case 7 以下. 猜测是因为 overhead.

## Machines

Using ceca 20.

Known to have GPU: ceca 20, ceca 22.

Connections are not very stable...

---

- Cannot log in ceca 22 by private keys. Confused.
