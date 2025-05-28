%%writefile cloud_fog_edge_advanced_demo.py
import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # For std dev calculation on empty lists

st.set_page_config(page_title="Advanced Cloud–Fog–Edge Simulation", layout="wide")
st.title("🌐 Advanced Cloud–Fog–Edge Computing Simulation")

# --- Session State Initialization ---
# This ensures data persists across reruns
if "results" not in st.session_state:
    st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}
if "current_load" not in st.session_state:
    st.session_state.current_load = {"Edge": 0.0, "Fog": 0.0, "Cloud": 0.0}
if "task_log" not in st.session_state:
    st.session_state.task_log = []
if "layer_queues" not in st.session_state:
    st.session_state.layer_queues = {"Edge": [], "Fog": [], "Cloud": []} # List of task_ids in queue
if "historical_avg_latency" not in st.session_state:
    st.session_state.historical_avg_latency = {"Edge": [], "Fog": [], "Cloud": []}

# --- Sidebar for Configuration ---
st.sidebar.header("⚙️ Simulation Configuration")

# Layer Capacities
st.sidebar.subheader("Layer Capacities (CPU Units)")
LAYER_CAPACITIES = {
    "Edge": st.sidebar.slider("Edge Capacity", 1.0, 20.0, 5.0, 0.5),
    "Fog": st.sidebar.slider("Fog Capacity", 5.0, 50.0, 15.0, 1.0),
    "Cloud": st.sidebar.slider("Cloud Capacity", 20.0, 100.0, 50.0, 2.0)
}

# Task Load Characteristics
st.sidebar.subheader("Task Characteristics")
MIN_TASK_LOAD = st.sidebar.slider("Min Task Load", 0.1, 5.0, 1.0, 0.1)
MAX_TASK_LOAD = st.sidebar.slider("Max Task Load", 1.0, 10.0, 5.0, 0.1)
MIN_TASK_DATA = st.sidebar.slider("Min Task Data Size (MB)", 0.1, 5.0, 0.5, 0.1)
MAX_TASK_DATA = st.sidebar.slider("Max Task Data Size (MB)", 1.0, 20.0, 5.0, 0.5)

# Latency Parameters
st.sidebar.subheader("Latency Parameters")
# Base processing time per unit load (lower = faster)
PROCESSING_FACTOR = {
    "Edge": st.sidebar.slider("Edge Proc. Factor", 0.01, 0.2, 0.05, 0.01),
    "Fog": st.sidebar.slider("Fog Proc. Factor", 0.02, 0.3, 0.08, 0.01),
    "Cloud": st.sidebar.slider("Cloud Proc. Factor", 0.005, 0.1, 0.02, 0.005) # Cloud is more efficient per unit
}
# Base network latency (fixed overhead + distance)
BASE_NETWORK_LATENCY = {
    "Edge": st.sidebar.slider("Edge Base Latency (sec)", 0.01, 0.1, 0.05, 0.01),
    "Fog": st.sidebar.slider("Fog Base Latency (sec)", 0.05, 0.3, 0.15, 0.01),
    "Cloud": st.sidebar.slider("Cloud Base Latency (sec)", 0.1, 0.8, 0.4, 0.01)
}
# Bandwidth for data transfer (MB/sec)
NETWORK_BANDWIDTH = {
    "Edge_Fog": st.sidebar.slider("Edge-Fog Bandwidth (MB/s)", 5.0, 50.0, 20.0, 1.0),
    "Fog_Cloud": st.sidebar.slider("Fog-Cloud Bandwidth (MB/s)", 10.0, 100.0, 50.0, 2.0)
}

SIMULATION_SPEED_MULTIPLIER = st.sidebar.slider("Simulation Speed Multiplier", 0.1, 5.0, 1.0, 0.1)

# Task Scheduling Policy
SCHEDULING_POLICY = st.sidebar.selectbox(
    "Scheduling Policy",
    ["Random", "Load-Aware", "Latency-Aware"]
)

# --- Task Simulation Function ---
def simulate_task(task_id):
    # Generate task characteristics
    task_load = random.uniform(MIN_TASK_LOAD, MAX_TASK_LOAD)
    task_data_size = random.uniform(MIN_TASK_DATA, MAX_TASK_DATA)

    # Initial layer assignment based on policy
    initial_layer_choice = None
    original_assigned_layer = None
    offloaded = False
    queue_time = 0.0

    if SCHEDULING_POLICY == "Random":
        possible_layers = ["Edge", "Fog", "Cloud"]
        weights = [0.5, 0.3, 0.2] # Default random weights
        initial_layer_choice = random.choices(possible_layers, weights=weights)[0]
    elif SCHEDULING_POLICY == "Load-Aware":
        # Choose the layer with the lowest current utilization
        utilization = {
            k: (st.session_state.current_load[k] / LAYER_CAPACITIES[k])
            for k in LAYER_CAPACITIES
        }
        # Find layers that can potentially handle the task without immediate overload
        eligible_layers = [
            layer for layer, load in st.session_state.current_load.items()
            if load + task_load <= LAYER_CAPACITIES[layer]
        ]
        if eligible_layers:
            initial_layer_choice = min(eligible_layers, key=lambda l: utilization[l])
        else: # All layers are overloaded, choose the one with least current load
            initial_layer_choice = min(LAYER_CAPACITIES.keys(), key=lambda l: st.session_state.current_load[l])

    elif SCHEDULING_POLICY == "Latency-Aware":
        estimated_latencies = {}
        for layer in LAYER_CAPACITIES.keys():
            # Estimate processing time
            estimated_proc_time = (task_load * PROCESSING_FACTOR[layer])
            
            # Estimate transmission time (if offloaded from Edge, consider Fog, then Cloud)
            transmission_time = 0.0
            if layer == "Fog":
                transmission_time += task_data_size / NETWORK_BANDWIDTH["Edge_Fog"]
            elif layer == "Cloud":
                # Assuming data might go Edge -> Fog -> Cloud, or directly Edge -> Cloud if direct path
                # For simplicity, let's consider Edge -> Fog -> Cloud for now
                transmission_time += (task_data_size / NETWORK_BANDWIDTH["Edge_Fog"]) + (task_data_size / NETWORK_BANDWIDTH["Fog_Cloud"])
            
            # Estimate queue time (very basic, assuming current queue processes instantly)
            # A more accurate queue time would simulate tasks processing in order
            estimated_queue_time = len(st.session_state.layer_queues[layer]) * (MIN_TASK_LOAD * PROCESSING_FACTOR[layer]) # Rough estimate

            estimated_latencies[layer] = estimated_proc_time + BASE_NETWORK_LATENCY[layer] + transmission_time + estimated_queue_time
        
        initial_layer_choice = min(estimated_latencies, key=estimated_latencies.get)

    original_assigned_layer = initial_layer_choice
    final_layer = initial_layer_choice

    # --- Queueing and Offloading Logic ---
    # Check if the chosen layer can immediately handle the task
    if st.session_state.current_load[final_layer] + task_load > LAYER_CAPACITIES[final_layer]:
        st.info(f"Task-{task_id} (Load: {task_load:.2f}) assigned to {final_layer} but it's busy. Adding to queue...")
        st.session_state.layer_queues[final_layer].append(task_id)
        
        # Simulate queue time: This is a simplification; in a real system, tasks
        # would wait for prior tasks in the queue to finish.
        # Here, we add a time penalty based on current queue length.
        queue_time = len(st.session_state.layer_queues[final_layer]) * random.uniform(0.05, 0.1) * SIMULATION_SPEED_MULTIPLIER
        time.sleep(queue_time)
        
        # After queue time, remove from queue and proceed
        if task_id in st.session_state.layer_queues[final_layer]:
            st.session_state.layer_queues[final_layer].remove(task_id)

        # Basic offloading attempt if queue is too long OR if policy specifically allows
        # For simplicity, if a task was queued, it's processed at that layer after queue time.
        # More advanced: try offloading to another layer if queue is too long.
        
    # Add task load to current layer load
    st.session_state.current_load[final_layer] += task_load

    # Calculate actual processing time based on load and processing factor
    processing_time = (task_load * PROCESSING_FACTOR[final_layer])

    # Calculate transmission time (if offloaded, which is simulated here by layer choice)
    transmission_time = 0.0
    if final_layer == "Fog" and original_assigned_layer == "Edge": # If task originated at Edge but got to Fog
         transmission_time = task_data_size / NETWORK_BANDWIDTH["Edge_Fog"]
         offloaded = True
    elif final_layer == "Cloud": # If task reached Cloud
        if original_assigned_layer == "Edge":
            # If from Edge, assume Edge -> Fog -> Cloud path for data
            transmission_time = (task_data_size / NETWORK_BANDWIDTH["Edge_Fog"]) + (task_data_size / NETWORK_BANDWIDTH["Fog_Cloud"])
            offloaded = True
        elif original_assigned_layer == "Fog":
            transmission_time = task_data_size / NETWORK_BANDWIDTH["Fog_Cloud"]
            offloaded = True

    total_latency = processing_time + BASE_NETWORK_LATENCY[final_layer] + transmission_time + queue_time

    # Simulate actual time spent (scaled by speed multiplier)
    time.sleep(total_latency * SIMULATION_SPEED_MULTIPLIER)

    # Decrement load after processing
    st.session_state.current_load[final_layer] -= task_load

    st.session_state.results[final_layer].append(total_latency)
    
    # Update historical average latency for this layer
    current_avg = sum(st.session_state.results[final_layer]) / len(st.session_state.results[final_layer])
    st.session_state.historical_avg_latency[final_layer].append(current_avg)


    # Log task details
    st.session_state.task_log.append({
        "Task ID": task_id,
        "Original Layer": original_assigned_layer,
        "Final Layer": final_layer,
        "Task Load": f"{task_load:.2f}",
        "Data Size (MB)": f"{task_data_size:.2f}",
        "Latency (sec)": f"{total_latency:.2f}",
        "Queue Time (sec)": f"{queue_time:.2f}",
        "Offloaded": "Yes" if offloaded else "No",
        "Timestamp": time.strftime("%H:%M:%S")
    })

    return final_layer, total_latency, task_load, offloaded, queue_time

# --- UI Layout ---
st.write("---") # Separator

# Placeholders for live updates
task_status_placeholder = st.empty()
live_utilization_placeholder = st.empty()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🚀 Generate Task"):
        task_id = len(st.session_state.task_log) + 1
        with st.spinner(f"Processing Task-{task_id}..."):
            layer, latency, task_load, offloaded, queue_time = simulate_task(task_id)
        
        offload_msg = "(Offloaded)" if offloaded else ""
        queue_msg = f"(Queued: {queue_time:.2f}s)" if queue_time > 0 else ""
        task_status_placeholder.success(f"✅ Task-{task_id} (Load: {task_load:.2f}, Data: {task_data_size:.2f}MB) processed by **{layer}** {offload_msg} {queue_msg} in {latency:.2f} seconds")
        
        # Update live utilization after each task
        with live_utilization_placeholder.container():
            st.subheader("⚡ Current Layer Load & Utilization")
            for layer_name, current_load in st.session_state.current_load.items():
                capacity = LAYER_CAPACITIES[layer_name]
                utilization_percent = (current_load / capacity) * 100 if capacity > 0 else 0
                st.write(f"**{layer_name}:** {current_load:.2f} / {capacity:.2f} CPU Units")
                st.progress(min(utilization_percent / 100, 1.0), text=f"{utilization_percent:.1f}% Utilization")

with col2:
    if st.button("🔄 Reset Simulation"):
        st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}
        st.session_state.current_load = {"Edge": 0.0, "Fog": 0.0, "Cloud": 0.0}
        st.session_state.task_log = []
        st.session_state.layer_queues = {"Edge": [], "Fog": [], "Cloud": []}
        st.session_state.historical_avg_latency = {"Edge": [], "Fog": [], "Cloud": []}
        st.success("Simulation reset!")
        task_status_placeholder.empty() # Clear previous task status
        live_utilization_placeholder.empty() # Clear live utilization
        st.rerun() # Rerun to clear the display

st.write("---")

st.subheader("📊 Task Summary & Performance")

# Create a DataFrame for task summary
summary_data = []
for k, v in st.session_state.results.items():
    if v:
        summary_data.append({
            "Layer": k,
            "Total Tasks": len(v),
            "Avg Latency (sec)": f"{sum(v)/len(v):.2f}",
            "Min Latency (sec)": f"{min(v):.2f}",
            "Max Latency (sec)": f"{max(v):.2f}",
            "Std Dev Latency (sec)": f"{np.std(v):.2f}" if len(v) > 1 else "N/A", # Use np.std for std dev
            "Total Load Processed": f"{sum(float(t['Task Load']) for t in st.session_state.task_log if t['Final Layer'] == k):.2f}"
        })
    else:
        summary_data.append({
            "Layer": k,
            "Total Tasks": 0,
            "Avg Latency (sec)": "N/A",
            "Min Latency (sec)": "N/A",
            "Max Latency (sec)": "N/A",
            "Std Dev Latency (sec)": "N/A",
            "Total Load Processed": "N/A"
        })
st.dataframe(pd.DataFrame(summary_data).set_index("Layer"), use_container_width=True)

st.subheader("📉 Average Latency Bar Chart")
fig, ax = plt.subplots(figsize=(10, 5))
avg_latency_vals = {k: (sum(v) / len(v)) if v else 0 for k, v in st.session_state.results.items()}
ax.bar(avg_latency_vals.keys(), avg_latency_vals.values(), color=["green", "orange", "blue"])
ax.set_ylabel("Avg Latency (sec)")
ax.set_ylim(0, max(1.5, max(avg_latency_vals.values()) * 1.1 if avg_latency_vals else 0.1)) # Dynamic y-limit
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

st.subheader("📈 Historical Average Latency Trend")
if any(len(v) > 0 for v in st.session_state.historical_avg_latency.values()):
    # Prepare data for plotting. Ensure all lists are of the same length for DataFrame.
    max_len = max(len(v) for v in st.session_state.historical_avg_latency.values())
    
    historical_df_data = {}
    for layer, data in st.session_state.historical_avg_latency.items():
        # Pad shorter lists with NaN to match max_len
        padded_data = data + [np.nan] * (max_len - len(data))
        historical_df_data[layer] = padded_data

    df_historical_avg = pd.DataFrame(historical_df_data)
    df_historical_avg.index.name = "Task # (Cumulative)"
    st.line_chart(df_historical_avg)
else:
    st.info("Generate some tasks to see the historical average latency trend.")


st.subheader("📋 Task Activity Log")
if st.session_state.task_log:
    # Reverse log to show most recent tasks first
    df_task_log = pd.DataFrame(st.session_state.task_log[::-1])
    st.dataframe(df_task_log, use_container_width=True)
else:
    st.info("No tasks have been processed yet.")

st.write("---")
st.markdown("Developed by Your Name/Team Name") # Optional: Add your name here!
