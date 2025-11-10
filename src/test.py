import random

random.seed(1)

posits = [[random.randrange(0,9) for _ in range(2)] for _ in range(100)]
store = 0
store &= 0xFFFFFFFFFFFFFFFF

total = 0
for (x, y) in posits:
    store = (store << 4) + x
    store = (store << 4) + y
    total += 1
print(store)
for _ in range(total):
    y = store & 0xF
    store = store >> 4
    x = store & 0xF
    store = store >> 4
    print(f"({x},{y}),")

print(posits)
