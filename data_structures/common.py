from heapq import heappush, heappop
import itertools

#########################################################################
# Search Stack
#########################################################################
'''
Implicit trade-off:
store more dead nodes in priority Q -> faster get and put times amortized O(1)
remove dead nodes in priority Q as they are found -> O(logN) get and put times
NOTE: if a lot more stack pops than heap pops -> may need GC (?)

Modified from python documentation for PriorityQ implementation:
https://docs.python.org/3/library/heapq.html
'''
class StackHeap:
    def __init__(self):
        self.pq = []  # list of entries arranged in a heap, ordered by priority
        self.sk = []  # list of entries arranged in a stack
        self.entry_finder = {}  # mapping of tasks to entries
        self.REMOVED = '<removed-task>'  # placeholder for a removed task
        self.counter = itertools.count()  # unique sequence count

    def __len__(self):
        return len(self.sk)

    def add(self, task, priority=0, front=False):
        """Add a new task or update the priority of an existing task. Lower priority is more important."""
        if task in self.entry_finder:
            self.remove_task(task, delete=False)
        count = next(self.counter)
        entry = [priority, count, task]
        self.entry_finder[task] = entry
        heappush(self.pq, entry)
        if front:
            self.sk = [entry] + self.sk
        else:
            self.sk.append(entry)

    def remove_task(self, task, delete):
        """Mark an existing task as REMOVED.  Raise KeyError if not found."""
        if delete:
            entry = self.entry_finder[task]
            self.sk.remove(entry)
            del self.entry_finder[task]
        else:
            entry = self.entry_finder.pop(task)
            self.sk.remove(entry)
            entry[-1] = self.REMOVED

    def get_task(self, method):
        if method == 'heap':
            priority, count, task = self.pq[0]
            while task is self.REMOVED:
                heappop(self.pq)
                priority, count, task = self.pq[0]
        elif method == 'stack':
            priority, count, task = self.sk[-1]
        else:
            assert False
        return task, priority

    def pop_task(self, method:str) -> (any, int):
        if method == 'heap':
            # Remove and return the lowest priority task.
            # Raise KeyError if empty.
            while self.pq:
                priority, count, task = heappop(self.pq)
                if task is not self.REMOVED:
                    self.remove_task(task, delete=True)
                    return task, priority
            raise KeyError('pop from an empty priority queue')
        elif method == 'stack':
            priority, count, task = self.sk[-1]
            self.remove_task(task, delete=False)
        elif method == 'queue':
            priority, count, task = self.sk[0]
            to_print = [s[2].tree_depth for s in self.sk]
            print('to_print', to_print)
            self.remove_task(task, delete=False)
        else:
            assert False

        return task, priority

#########################################################################
# Double Dictionary
#########################################################################
class DoubleDict():
    def __init__(self):
        self.l2r = {}
        self.r2l = {}

    def __len__(self):
        return len(self.l2r)
    
    def __str__(self):
        return f'l2r: {self.l2r} \t\t r2l: {self.r2l}'

    def add_lr(self, l, r):
        if l not in self.l2r:
            self.l2r[l] = set()
        if r not in self.r2l:
            self.r2l[r] = set()
        self.l2r[l].add(r)
        self.r2l[r].add(l)

    def get_l2r(self, l):
        if l not in self.l2r:
            return set()
        else:
            return self.l2r[l]

    def get_r2l(self, r):
        if r not in self.r2l:
            return set()
        else:
            return self.r2l[r]

