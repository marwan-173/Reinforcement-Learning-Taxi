from src.planning import Planner

# Test the Planner priority queue.

planner = Planner()

planner.add(1.5, 10, 2)
planner.add(3.0, 20, 1)
planner.add(2.0, 30, 5)

priority, state, action = planner.pop()

assert priority == 3.0
assert state == 20
assert action == 1

assert planner.size() == 2

planner.clear()

assert planner.is_empty()

print("Planning test passed.")