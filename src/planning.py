import heapq

class Planner:

    def __init__(self):      #Initializing prioritized sweeping planner.
        self.priority_queue = []

    def add(self, priority, state, action):      #Add a state-action pair to the priority queue.
        heapq.heappush(self.priority_queue, (-priority, state, action))

    def pop(self):      #Remove the highest priority state-action pair.
        if self.is_empty():
            return None

        priority, state, action = heapq.heappop(self.priority_queue)
        return -priority, state, action

    def is_empty(self):      #Check whether the priority queue is empty.
        return len(self.priority_queue) == 0

    def clear(self):      #Remove all queued transitions.
        self.priority_queue.clear()

    def size(self):      #Return the number of queued transitions.
        return len(self.priority_queue)