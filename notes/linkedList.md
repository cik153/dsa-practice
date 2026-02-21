# LinkedList

## What it is
**ArrayList**는 데이터를 “노드(node)” 단위로 저장하고, 각 노드가 다음 노드를 가리키는(링크하는) 자료구조
- 새로운 노드를 삽입시에, 그 삽입되는 곳의 앞 노드의 link에 저장된 주소를 새로운 노드의 link에 넣어 삽입될 곳의 뒷 노드를 연결시켜주고, 삽입되는 곳의 앞 노드의 link에 새로운 노드의 주소값을 넣어 연결시켜준다.
- newNode.link <- preNode.link
- preNode.link <- newNode

> Typical names by language:
> - Java: `ArrayList`
> - C++: `std::vector`
> - C#: `List<T>`
> - Python: `list`

---

## Core operations & Big-O
| Operation | Average Time | Notes |
|---|---:|---|
| `get(i)` / `set(i)` | O(1) | Random access |
| `append(x)` | Amortized O(1) | Usually constant, occasional resize |
| `insert(i, x)` | O(n) | Shift elements to the right |
| `remove(i)` | O(n) | Shift elements to the left |
| `remove(x)` | O(n) | Need to find `x` first (linear search) |
| `contains(x)` | O(n) | Linear search |
| iteration | O(n) | Sequential scan |

**Key point:**  
Append is **amortized O(1)** because resizing happens rarely, but when it happens it costs O(n).

---

## How resizing works (important)
When capacity is full and we append:
1) Allocate a new array (usually **2x** capacity)  
2) Copy all elements to the new array (**O(n)**)  
3) Free the old array  
4) Insert the new element

Because this expensive copy happens occasionally, average append cost becomes **amortized O(1)**.

---

## When ArrayList is a good choice
✅ Use ArrayList when you need:
- Frequent **index access** (`get(i)`)
- Mostly **append at the end**
- Iteration speed (cache-friendly contiguous memory)

❌ Avoid ArrayList when you need:
- Frequent **insert/remove at the front or middle**
- Many deletions in arbitrary positions

---

## Common pitfalls
- **Insert/Delete in the middle is slow** due to shifting
- **Capacity vs Size**
  - Size: number of elements
  - Capacity: allocated space (can be larger than size)
- Removing many items inside a loop can cause **O(n^2)** if done poorly

---

## Patterns / Tips
### 1) Reserve capacity (if the language supports it)
If you know approximate item count, reserve to reduce reallocations:
- C++: `vector.reserve(n)`
- C#: `new List<T>(capacity)` or `EnsureCapacity(n)`

### 2) Remove many elements efficiently
If removing based on a condition, prefer a "filter" approach:
- Keep elements you want, overwrite in-place (two-pointer), then shrink.

---

## Practice problems (add links as you solve)
- [ ] (BOJ) ________
- [ ] (Programmers) ________
- [ ] (LeetCode) ________

### Solutions (code links)
- [Soulutions](../solutions/)
