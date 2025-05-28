import streamlit as st
import time
import random
import matplotlib.pyplot as plt
import pandas as pd # Import pandas

st.set_page_config(page_title="Cloud-Fog-Edge Simulation", layout="centered")
st.title("🌐 Cloud–Fog–Edge Computing Simulation")

# --- Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}
if "current_load" not in st.session_state:
    st.session_state.current_load = {"Edge": 0.0, "Fog": 0.0, "Cloud": 0.0} # Initialize as floats
if "task_log" not in st.session_state: # New: to store details of each task
    st.session_state.task_log = []

# Define capacities
LAYER_CAPACITIES = {"Edge": 5.0, "Fog": 10.0, "Cloud": 20.0}

def simulate_task(task_id):
    task_load = random.uniform(1.0, 5.0)

    possible_layers = ["Edge", "Fog", "Cloud"]
    # Adjust weights to favor Edge slightly more for initial assignment
    weights = [0.55, 0.30, 0.15]
    assigned_layer = random.choices(possible_layers, weights=weights)[0]

    original_assigned_layer = assigned_layer
    offloaded = False
    final_layer = assigned_layer # Will be updated if offloaded

    # --- Simple Offloading Logic ---
    # This logic needs to be more robust for production, e.g., iterating through layers
    # or using a more sophisticated algorithm.
    if st.session_state.current_load[assigned_layer] + task_load > LAYER_CAPACITIES[assigned_layer]:
        st.warning(f"Task-{task_id} (Load: {task_load:.2f}) initially assigned to {assigned_layer} but overloaded.")
        
        # Try offloading to the next available layer with enough capacity
        if assigned_layer == "Edge":
            if st.session_state.current_load["Fog"] + task_load <= LAYER_CAPACITIES["Fog"]:
                final_layer = "Fog"
                offloaded = True
            elif st.session_state.current_load["Cloud"] + task_load <= LAYER_CAPACITIES["Cloud"]:
                final_layer = "Cloud"
                offloaded = True
        elif assigned_layer == "Fog":
            if st.session_state.current_load["Cloud"] + task_load <= LAYER_CAPACITIES["Cloud"]:
                final_layer = "Cloud"
                offloaded = True
        
        if not offloaded:
            # If no offloading option, the task is "stuck" or queued at the original layer.
            # For this demo, we'll still process it at the original layer, but acknowledge overload.
            st.error(f"Task-{task_id} (Load: {task_load:.2f}) could not be offloaded. Processing on {original_assigned_layer} (OVERLOADED).")
            final_layer = original_assigned_layer # Process at original despite overload
            
    # Update current load for the final chosen layer
    st.session_state.current_load[final_layer] += task_load

    # Calculate latency based on layer and task load
    # More sophisticated latency model: (Task Load / Layer Capacity) + Base Network Latency
    base_network_latency = {
        "Edge": random.uniform(0.05, 0.15),
        "Fog": random.uniform(0.15, 0.3),
        "Cloud": random.uniform(0.3, 0.6)
    }
    
    # Simulate processing time based on load and capacity
    # Add a small base processing time to avoid 0 latency for low loads
    processing_time_factor = random.uniform(0.05, 0.1) # How efficient layer is per unit load
    latency = (task_load / LAYER_CAPACITIES[final_layer]) * processing_time_factor + base_network_latency[final_layer]
    
    # Cap latency to avoid extremely long waits in demo
    latency = min(latency, 2.5) 

    # Simulate actual time spent
    time.sleep(latency)

    # Decrement load after processing
    st.session_state.current_load[final_layer] -= task_load

    st.session_state.results[final_layer].append(latency)
    
    # Log task details
    st.session_state.task_log.append({
        "Task ID": task_id,
        "Original Layer": original_assigned_layer,
        "Final Layer": final_layer,
        "Task Load": f"{task_load:.2f}",
        "Latency (sec)": f"{latency:.2f}",
        "Offloaded": "Yes" if offloaded else "No",
        "Timestamp": time.strftime("%H:%M:%S")
    })

    return final_layer, latency, task_load, offloaded

# --- UI Layout ---
col1, col2 = st.columns([1, 3])

with col1:
    if st.button("🚀 Generate Task"):
        task_id = len(sum(st.session_state.results.values(), [])) + 1
        with st.spinner(f"Processing Task-{task_id}..."):
            layer, latency, task_load, offloaded = simulate_task(task_id)
        offload_msg = "(Offloaded)" if offloaded else ""
        st.success(f"✅ Task-{task_id} (Load: {task_load:.2f}) processed by **{layer}** {offload_msg} in {latency:.2f} seconds")

    if st.button("🔄 Reset Simulation"):
        st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}
        st.session_state.current_load = {"Edge": 0.0, "Fog": 0.0, "Cloud": 0.0}
        st.session_state.task_log = []
        st.success("Simulation reset!")
        st.rerun() # Rerun to clear the display

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
            "Std Dev Latency (sec)": f"{pd.Series(v).std():.2f}" if len(v) > 1 else "N/A"
        })
    else:
        summary_data.append({
            "Layer": k,
            "Total Tasks": 0,
            "Avg Latency (sec)": "N/A",
            "Min Latency (sec)": "N/A",
            "Max Latency (sec)": "N/A",
            "Std Dev Latency (sec)": "N/A"
        })
st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

st.subheader("📉 Average Latency Chart")
fig, ax = plt.subplots(figsize=(8, 4))
avg_latency = {k: (sum(v) / len(v)) if v else 0 for k, v in st.session_state.results.items()}
ax.bar(avg_latency.keys(), avg_latency.values(), color=["green", "orange", "blue"])
ax.set_ylabel("Avg Latency (sec)")
ax.set_ylim(0, max(1.5, max(avg_latency.values()) * 1.1 if avg_latency else 0.1)) # Dynamic y-limit
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)


st.subheader("⚡ Current Layer Load")
# Create a DataFrame for current load
load_data = []
for k, v in st.session_state.current_load.items():
    load_data.append({
        "Layer": k,
        "Current Load": f"{v:.2f}",
        "Capacity": f"{LAYER_CAPACITIES[k]:.2f}",
        "Utilization (%)": f"{((v / LAYER_CAPACITIES[k]) * 100):.2f}%" if LAYER_CAPACITIES[k] > 0 else "N/A"
    })
st.dataframe(pd.DataFrame(load_data), use_container_width=True)

# New: Historical Latency Trend
st.subheader("📈 Historical Average Latency")
if any(len(v) > 0 for v in st.session_state.results.values()):
    # Convert results to a format suitable for line chart
    # This example simply plots the current average each time, a more detailed history
    # would store snapshots over time.
    chart_data = {
        "Task Count": list(range(1, len(st.session_state.task_log) + 1))
    }
    
    for layer in st.session_state.results.keys():
        current_layer_latencies = st.session_state.results[layer]
        if current_layer_latencies:
            chart_data[layer] = [current_layer_latencies[i] for i in range(len(current_layer_latencies))]
            # This is a simple way; for true historical, you'd store avg per task.
            # For demonstration, just plot current avg for each run:
            # chart_data[layer] = [sum(current_layer_latencies[:i+1])/(i+1) for i in range(len(current_layer_latencies))]
            
            # A simpler approach for the line chart: plot individual latencies
            # For a true "historical average trend", you'd need to store the average
            # after each task or at regular intervals.
            # Let's use `task_log` for a combined latency view
            
    # For a historical *average* trend, you'd need to calculate averages
    # for each layer as tasks are processed over time.
    # A simpler but effective visualization is a line chart of individual task latencies over time.
    
    # If you want true historical average per layer:
    historical_avg_data = {}
    for layer in st.session_state.results.keys():
        historical_avg_data[layer] = []
        cumulative_sum = 0
        for i, latency in enumerate(st.session_state.results[layer]):
            cumulative_sum += latency
            historical_avg_data[layer].append(cumulative_sum / (i + 1))

    # Create a DataFrame for the historical average
    max_len = max(len(v) for v in historical_avg_data.values()) if historical_avg_data else 0
    df_historical = pd.DataFrame(index=range(max_len))
    for layer, data in historical_avg_data.items():
        df_historical[layer] = pd.Series(data).reindex(range(max_len), fill_value=None) # Use None for missing values
    
    st.line_chart(df_historical)
else:
    st.info("Generate some tasks to see the historical average latency.")

st.subheader("📋 Task Activity Log")
if st.session_state.task_log:
    # Reverse log to show most recent tasks first
    df_task_log = pd.DataFrame(st.session_state.task_log[::-1])
    st.dataframe(df_task_log, use_container_width=True)
else:
    st.info("No tasks have been processed yet.")
