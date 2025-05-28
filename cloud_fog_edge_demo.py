import streamlit as st
import time
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud-Fog-Edge Simulation", layout="centered")
st.title("🌐 Cloud–Fog–Edge Computing Simulation")

if "results" not in st.session_state:
    st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}
if "current_load" not in st.session_state: # To track current load
    st.session_state.current_load = {"Edge": 0, "Fog": 0, "Cloud": 0}

# Define capacities (example values)
LAYER_CAPACITIES = {"Edge": 5.0, "Fog": 10.0, "Cloud": 20.0} # Units could be "computation units per second"

def simulate_task(task_id):
    # Assign a random computational load to the task
    task_load = random.uniform(1.0, 5.0) # Example: task requires 1 to 5 units of computation

    # Initial random assignment (can be replaced by smarter logic later)
    possible_layers = ["Edge", "Fog", "Cloud"]
    weights = [0.5, 0.3, 0.2]
    assigned_layer = random.choices(possible_layers, weights=weights)[0]

    # --- New Logic: Check capacity and potentially offload ---
    original_assigned_layer = assigned_layer
    offloaded = False

    # A simple greedy offloading strategy: try to find an available layer
    # This is a very basic example; a real system would have more complex logic.
    if st.session_state.current_load[assigned_layer] + task_load > LAYER_CAPACITIES[assigned_layer]:
        st.warning(f"Task-{task_id} (Load: {task_load:.2f}) initially assigned to {assigned_layer} but overloaded. Attempting offload...")
        # Try to offload to the next layer (Fog -> Cloud, Edge -> Fog or Cloud)
        if assigned_layer == "Edge":
            if st.session_state.current_load["Fog"] + task_load <= LAYER_CAPACITIES["Fog"]:
                assigned_layer = "Fog"
                offloaded = True
            elif st.session_state.current_load["Cloud"] + task_load <= LAYER_CAPACITIES["Cloud"]:
                assigned_layer = "Cloud"
                offloaded = True
        elif assigned_layer == "Fog":
            if st.session_state.current_load["Cloud"] + task_load <= LAYER_CAPACITIES["Cloud"]:
                assigned_layer = "Cloud"
                offloaded = True
        if not offloaded:
            # If offloading failed, assign to the original layer and just add to its queue/load
            # Or you could mark it as "failed" or "queued"
            st.error(f"Task-{task_id} could not be offloaded from {original_assigned_layer}. Processing there anyway (or marking as failed).")
            assigned_layer = original_assigned_layer # Stick to original if no better option found

    # Update current load for the chosen layer
    st.session_state.current_load[assigned_layer] += task_load

    # Calculate latency based on layer and task load
    base_latency_per_unit = {
        "Edge": random.uniform(0.02, 0.05), # faster per unit load
        "Fog": random.uniform(0.04, 0.08),  # medium per unit load
        "Cloud": random.uniform(0.03, 0.06) # faster per unit than Fog for raw computation, but has higher base network latency
    }
    # Add a base network latency for each layer
    network_latency = {
        "Edge": random.uniform(0.05, 0.1),
        "Fog": random.uniform(0.1, 0.2),
        "Cloud": random.uniform(0.2, 0.5)
    }

    latency = (task_load / LAYER_CAPACITIES[assigned_layer]) + network_latency[assigned_layer] # Simplified model
    # Or more directly:
    # latency = task_load * base_latency_per_unit[assigned_layer] + network_latency[assigned_layer]


    # Simulate actual time spent
    time.sleep(min(latency, 2.0)) # Cap sleep to avoid very long waits in demo

    # Decrement load after processing (assuming instant release)
    st.session_state.current_load[assigned_layer] -= task_load

    st.session_state.results[assigned_layer].append(latency)
    return assigned_layer, latency, task_load, offloaded

if st.button("🚀 Generate Task"):
    task_id = len(sum(st.session_state.results.values(), [])) + 1
    with st.spinner("Processing..."):
        layer, latency, task_load, offloaded = simulate_task(task_id)
    offload_msg = "(Offloaded)" if offloaded else ""
    st.success(f"✅ Task-{task_id} (Load: {task_load:.2f}) processed by **{layer}** {offload_msg} in {latency:.2f} seconds")

st.subheader("📊 Task Summary")
st.write({k: f"{len(v)} tasks, avg {sum(v)/len(v):.2f} sec" if v else "No tasks yet"
          for k, v in st.session_state.results.items()})

st.subheader("📉 Average Latency Chart")
fig, ax = plt.subplots()
avg_latency = {k: (sum(v) / len(v)) if v else 0 for k, v in st.session_state.results.items()}
ax.bar(avg_latency.keys(), avg_latency.values(), color=["green", "orange", "blue"])
ax.set_ylabel("Avg Latency (sec)")
ax.set_ylim(0, 1.5)
ax.grid(True)
st.pyplot(fig)

st.subheader("Current Layer Load")
st.write({k: f"{v:.2f} / {LAYER_CAPACITIES[k]:.2f}" for k, v in st.session_state.current_load.items()})
