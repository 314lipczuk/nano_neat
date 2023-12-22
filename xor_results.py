import os
basepath = 'tmp/XOR'
first = list(os.walk(basepath))[0]
dirs = first[1]
all_gens = {}
for d in dirs:
    max = 0
    for _, _, files in os.walk(f"{basepath}/{d}"):
        for f in files:
            if f.endswith(".pkl"):
                name = f.split(".")[0]
                gen = int(name.split("_")[1])
                if gen > max:
                    max = gen
    print(f"{d} {max}")
    all_gens[d] = max

