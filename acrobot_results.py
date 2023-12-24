import os
import math

def calculate_std_dev(list, n ,mean): return math.sqrt(sum( [(i-mean)**2 for i in list] ) /n )

basepath = 'tmp/ACROBOT'
first = list(os.walk(basepath))[0]
dirs = first[1]
average_generations = 0
count = 0
data = {
    "generations":[],
    "connections":[],
    "nodes":[],
}
average_nodes= 0
average_connections= 0
for d in dirs:
    max = 0
    for _, _, files in os.walk(f"{basepath}/{d}"):
        for f in files:
            if f.endswith(".txt"):
                with open(f"{basepath}/{d}/{f}") as file:
                    line = file.readline().split(' ')
                    for i,e in enumerate(["generations", "nodes", "connections"]):
                        data[e].append(int(line[i]))

count = len(data["generations"])
for k in ["generations", "nodes", "connections"]: 
    data["average"+k] = sum(data[k]) / count
    data["std_dev_"+k] = calculate_std_dev(data[k], count, data["average"+k])

data['connections'] = 0
data['nodes'] = 0
data['generations'] = 0
print(data, 'count', count)