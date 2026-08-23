# quick_test_graph.py — throwaway
from langgraph.types import Command
from graph import compiled

config = {"configurable": {"thread_id": "test-1"}}

# Step 1: run until the interrupt fires
for event in compiled.stream({"topic": "quantum computing"}, config, stream_mode="updates"):
    print("EVENT:", event)

# Step 2: inspect the plan, then resume approving everything
state = compiled.get_state(config)
print("\nPAUSED. Plan proposed:", state.values["plan"])

for event in compiled.stream(
    Command(resume={"approved": list(state.values["plan"]["assignments"].keys())}),
    config,
    stream_mode="updates",
):
    print("EVENT:", event)

# Step 3: check the final report
final = compiled.get_state(config)
print("\n--- REPORT ---\n")
print(final.values["report"])