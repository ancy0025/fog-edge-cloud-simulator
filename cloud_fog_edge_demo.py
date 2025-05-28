import streamlit as st
import time
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fog-Edge-Cloud Simulation", layout="centered")
st.title("🌐 Cloud–Fog–Edge Computing Simulation")

if "results" not in st.session_state:
    st.session_state.results = {"Edge": [], "Fog": [], "Cloud": []}

def simulate_task(task_id):
    layer = random.choices(["Edge", "Fog", "Cloud"], weights=[0.5, 0.3, 0.2])[0]
    if layer == "Edge":
        latency = random.uniform(0.1, 0.3)
    elif layer == "Fog":
        latency = random.uniform(0.4, 0.7)
    else:
        latency = random.uniform(0.8, 1.3)
    time.sleep(latency)
    st.session_state.results[layer].append(latency)
    return layer, latency

if st.button("🚀 Generate Task"):
    task_id = len(sum(st.session_state.results.values(), [])) + 1
    with st.spinner("Processing..."):
        layer, latency = simulate_task(task_id)
    st.success(f"✅ Task-{task_id} processed by **{layer}** in {latency:.2f} seconds")

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
