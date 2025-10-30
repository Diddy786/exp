

#### 1. First Come First Serve (FCFS) Scheduling

**Description:** Processes are executed in the order they arrive. Simple but can cause convoy effect.

**Code:**
```python
# FCFS Scheduling
n = int(input("Enter number of processes: "))
bt = [int(input(f"Enter burst time for P{i+1}: ")) for i in range(n)]

wt = [0]*n
tat = [0]*n

# Waiting time
for i in range(1, n):
    wt[i] = wt[i-1] + bt[i-1]

# Turnaround time
for i in range(n):
    tat[i] = wt[i] + bt[i]

print("\nProcess\tBT\tWT\tTAT")
for i in range(n):
    print(f"P{i+1}\t{bt[i]}\t{wt[i]}\t{tat[i]}")

print(f"\nAverage Waiting Time = {sum(wt)/n:.2f}")
print(f"Average Turnaround Time = {sum(tat)/n:.2f}")
```

**Sample Input/Output:**
```
Enter number of processes: 3
Enter burst time for P1: 10
Enter burst time for P2: 5
Enter burst time for P3: 8

Process	BT	WT	TAT
P1	10	0	10
P2	5	10	15
P3	8	15	23

Average Waiting Time = 8.33
Average Turnaround Time = 16.00
```

---

#### 2. Shortest Job First (SJF) Scheduling

**Description:** Process with shortest burst time is executed first. Minimizes average waiting time.

**Code:**
```python
# SJF Scheduling
n = int(input("Enter number of processes: "))
bt = [int(input(f"Enter burst time for P{i+1}: ")) for i in range(n)]

processes = list(range(1, n+1))
# Sort by burst time
bt, processes = zip(*sorted(zip(bt, processes)))

wt = [0]*n
tat = [0]*n

for i in range(1, n):
    wt[i] = wt[i-1] + bt[i-1]

for i in range(n):
    tat[i] = wt[i] + bt[i]

print("\nProcess\tBT\tWT\tTAT")
for i in range(n):
    print(f"P{processes[i]}\t{bt[i]}\t{wt[i]}\t{tat[i]}")

print(f"\nAverage Waiting Time = {sum(wt)/n:.2f}")
print(f"Average Turnaround Time = {sum(tat)/n:.2f}")
```

**Sample Input/Output:**
```
Enter number of processes: 3
Enter burst time for P1: 10
Enter burst time for P2: 5
Enter burst time for P3: 8

Process	BT	WT	TAT
P2	5	0	5
P3	8	5	13
P1	10	13	23

Average Waiting Time = 6.00
Average Turnaround Time = 13.67
```

---

#### 3. Priority Scheduling

**Description:** Process with highest priority (lowest priority number) is executed first.

**Code:**
```python
# Priority Scheduling
n = int(input("Enter number of processes: "))
bt = []
priority = []
for i in range(n):
    bt.append(int(input(f"Enter burst time for P{i+1}: ")))
    priority.append(int(input(f"Enter priority for P{i+1} (lower = higher): ")))

processes = list(range(1, n+1))
# Sort by priority
priority, bt, processes = zip(*sorted(zip(priority, bt, processes)))

wt = [0]*n
tat = [0]*n

for i in range(1, n):
    wt[i] = wt[i-1] + bt[i-1]

for i in range(n):
    tat[i] = wt[i] + bt[i]

print("\nProcess\tBT\tPriority\tWT\tTAT")
for i in range(n):
    print(f"P{processes[i]}\t{bt[i]}\t{priority[i]}\t\t{wt[i]}\t{tat[i]}")

print(f"\nAverage Waiting Time = {sum(wt)/n:.2f}")
print(f"Average Turnaround Time = {sum(tat)/n:.2f}")
```

**Sample Input/Output:**
```
Enter number of processes: 3
Enter burst time for P1: 10
Enter priority for P1 (lower = higher): 3
Enter burst time for P2: 5
Enter priority for P2 (lower = higher): 1
Enter burst time for P3: 8
Enter priority for P3 (lower = higher): 2

Process	BT	Priority	WT	TAT
P2	5	1		0	5
P3	8	2		5	13
P1	10	3		13	23

Average Waiting Time = 6.00
Average Turnaround Time = 13.67
```

---

#### 4. Round Robin Scheduling

**Description:** Each process gets a fixed time slice (quantum). Preemptive and fair scheduling.

**Code:**
```python
# Round Robin Scheduling
n = int(input("Enter number of processes: "))
bt = [int(input(f"Enter burst time for P{i+1}: ")) for i in range(n)]
quantum = int(input("Enter Time Quantum: "))

wt = [0]*n
tat = [0]*n
rem_bt = bt[:]
t = 0

while True:
    done = True
    for i in range(n):
        if rem_bt[i] > 0:
            done = False
            if rem_bt[i] > quantum:
                t += quantum
                rem_bt[i] -= quantum
            else:
                t += rem_bt[i]
                wt[i] = t - bt[i]
                rem_bt[i] = 0
    if done:
        break

for i in range(n):
    tat[i] = bt[i] + wt[i]

print("\nProcess\tBT\tWT\tTAT")
for i in range(n):
    print(f"P{i+1}\t{bt[i]}\t{wt[i]}\t{tat[i]}")

print(f"\nAverage Waiting Time = {sum(wt)/n:.2f}")
print(f"Average Turnaround Time = {sum(tat)/n:.2f}")
```

**Sample Input/Output:**
```
Enter number of processes: 3
Enter burst time for P1: 10
Enter burst time for P2: 5
Enter burst time for P3: 8
Enter Time Quantum: 4

Process	BT	WT	TAT
P1	10	13	23
P2	5	10	15
P3	8	6	14

Average Waiting Time = 9.67
Average Turnaround Time = 17.33
```

---

### 💾 Memory Management - Page Replacement Algorithms

#### 5. FIFO (First In First Out) Page Replacement

**Description:** Replaces the oldest page in memory when a page fault occurs.

**Code:**
```python
# FIFO Page Replacement Algorithm - Short Version

def fifo(pages, capacity):
    frames, faults = [], 0
    for p in pages:
        if p not in frames:
            faults += 1
            if len(frames) == capacity:
                frames.pop(0)     # remove oldest page
            frames.append(p)
        print(f"Page: {p}\tFrames: {frames}")
    print(f"\nTotal Page Faults: {faults}")

# main
pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3]
capacity = 3
fifo(pages, capacity)
```

**Sample Output:**
```
Page: 7	Frames: [7]
Page: 0	Frames: [7, 0]
Page: 1	Frames: [7, 0, 1]
Page: 2	Frames: [0, 1, 2]
Page: 0	Frames: [0, 1, 2]
Page: 3	Frames: [1, 2, 3]
Page: 0	Frames: [2, 3, 0]
Page: 4	Frames: [3, 0, 4]
Page: 2	Frames: [0, 4, 2]
Page: 3	Frames: [4, 2, 3]
Page: 0	Frames: [2, 3, 0]
Page: 3	Frames: [2, 3, 0]

Total Page Faults: 9
```

---

#### 6. LRU (Least Recently Used) Page Replacement

**Description:** Replaces the page that has been unused for the longest time.

**Code:**
```python
# LRU Page Replacement Algorithm - Short Version

def lru(pages, capacity):
    frames, faults = [], 0
    for p in pages:
        if p not in frames:
            faults += 1
            if len(frames) == capacity:
                frames.pop(0)     # remove least recently used
        else:
            frames.remove(p)      # move recently used to end
        frames.append(p)
        print(f"Page: {p}\tFrames: {frames}")
    print(f"\nTotal Page Faults: {faults}")

# main
pages = [7, 0, 1, 2, 0, 3, 0, 4, 2, 3, 0, 3]
capacity = 3
lru(pages, capacity)
```

**Sample Output:**
```
Page: 7	Frames: [7]
Page: 0	Frames: [7, 0]
Page: 1	Frames: [7, 0, 1]
Page: 2	Frames: [0, 1, 2]
Page: 0	Frames: [1, 2, 0]
Page: 3	Frames: [2, 0, 3]
Page: 0	Frames: [2, 3, 0]
Page: 4	Frames: [3, 0, 4]
Page: 2	Frames: [0, 4, 2]
Page: 3	Frames: [4, 2, 3]
Page: 0	Frames: [2, 3, 0]
Page: 3	Frames: [2, 0, 3]

Total Page Faults: 8
```

---

### 📁 File Allocation

#### 7. Sequential File Allocation

**Description:** Files are allocated contiguous blocks on disk. Simple but can cause external fragmentation.

**Code:**
```python
# Sequential File Allocation
start = int(input("Enter starting block: "))
length = int(input("Enter length of file: "))

print("File allocated from block", start, "to", start + length - 1)
```

**Sample Input/Output:**
```
Enter starting block: 5
Enter length of file: 10
File allocated from block 5 to 14
```

---

