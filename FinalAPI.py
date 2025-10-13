import simpy
import numpy as np
import random
import collections
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import json
import asyncio
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import logging
from typing import List, Dict, Any
import os
import csv
import io
import base64
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simulation parameters
SIMULATION_TIME = 1440  # Total simulation time
NUM_TRACKS = 22           # Number of available tracks at the station
TRAIN_INTERVAL = 6      # Average time between train arrivals

# DQN parameters
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 10000
BATCH_SIZE = 64
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
TARGET_UPDATE_FREQ = 100

# Global variables for WebSocket communication
active_connections = []
# Use thread-safe data structures
simulation_data = {
    "throughput_history": [],
    "avg_wait_history": [],
    "exploration_history": [],
    "loss_history": [],
    "train_events": [],
    "simulation_status": "not_started",  # Add simulation status
    "summary_stats": {}
}
# Lock for thread-safe access to simulation_data
data_lock = threading.Lock()

# Flag to control if simulation is running
simulation_thread = None
simulation_running = False

class TrainStation:
    def __init__(self, env, num_tracks):
        self.env = env
        self.tracks = simpy.Resource(env, num_tracks)
        self.waiting_trains = []       # Trains waiting to enter the station
        self.departure_queue = []      # Trains ready to depart
        self.throughput = 0
        self.avg_wait_time = 0
        self.total_wait_time = 0
        self.throughput_history = []
        self.avg_wait_history = []

    def update_metrics(self, wait_time):
        self.total_wait_time += wait_time
        self.throughput += 1
        self.avg_wait_time = self.total_wait_time / self.throughput
        self.throughput_history.append(self.throughput)
        self.avg_wait_history.append(self.avg_wait_time)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX
        self.exploration_history = []
        self.loss_history = []
        self.reward_history = []

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_size)
        else:
            q_values = self.model.predict(state, verbose=0)
            action = np.argmax(q_values[0])
        self.exploration_history.append(self.exploration_rate)
        return action

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return 0
        batch = random.sample(self.memory, BATCH_SIZE)
        states = np.array([experience[0][0] for experience in batch])
        next_states = np.array([experience[3][0] for experience in batch])
        current_q = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                current_q[i][action] = reward
            else:
                current_q[i][action] = reward + GAMMA * np.amax(next_q[i])

        history = self.model.fit(states, current_q, batch_size=BATCH_SIZE, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)

        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate * EXPLORATION_DECAY)
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        if not name.endswith('.weights.h5'):
            name += '.weights.h5'
        self.model.save_weights(name)

class Train:
    def __init__(self, env, train_id, arrival_time):
        self.env = env
        self.id = train_id
        self.arrival_time = arrival_time
        self.departure_time = None
        self.wait_time = 0
        # Fixed dwell time = 5 minutes for every train
        self.dwell_time = 5

async def broadcast_data():
    """Broadcast simulation data to all connected WebSocket clients"""
    while True:
        if active_connections:
            try:
                # Create a copy of the data to avoid race conditions
                with data_lock:
                    data = {
                        "throughput_history": simulation_data["throughput_history"][-100:],
                        "avg_wait_history": simulation_data["avg_wait_history"][-100:],
                        "exploration_history": simulation_data["exploration_history"][-100:],
                        "loss_history": simulation_data["loss_history"][-100:],
                        "train_events": simulation_data["train_events"][-20:],  # Last 20 events
                        "simulation_status": simulation_data["simulation_status"],
                        "summary_stats": simulation_data["summary_stats"]
                    }
                
                for connection in active_connections:
                    try:
                        await connection.send_text(json.dumps(data))
                    except Exception as e:
                        logger.error(f"Error sending to connection: {e}")
                        if connection in active_connections:
                            active_connections.remove(connection)
            except Exception as e:
                logger.error(f"Error preparing broadcast data: {e}")
        
        await asyncio.sleep(1)  # Update every second

def train_generator(env, station, train_counter):
    # Initialize next arrival time
    env.next_arrival = env.now + np.random.exponential(TRAIN_INTERVAL)
    
    while True:
        # Wait until next arrival time
        yield env.timeout(env.next_arrival - env.now)
        
        # Create the train
        train_id = next(train_counter)
        train = Train(env, train_id, env.now)
        station.waiting_trains.append(train)
        
        # Add to simulation data for broadcasting
        with data_lock:
            event = {
                "type": "arrival",
                "train_id": train_id,
                "time": env.now,
                "wait_time": 0,
                "throughput": station.throughput
            }
            simulation_data["train_events"].append(event)
        
        logger.info(f"Train {train_id} arrived at time {env.now}")
        
        # Schedule next arrival and handling simulation issue.
        if env.now + np.random.exponential(TRAIN_INTERVAL) <= SIMULATION_TIME:
            env.next_arrival = env.now + np.random.exponential(TRAIN_INTERVAL)
        else:
            env.next_arrival = float('inf')  # No more arrivals
            break

def state_representation(station, env):
    waiting_count = len(station.waiting_trains)
    at_station_count = station.tracks.count
    free_tracks = NUM_TRACKS - at_station_count
    ready_to_depart = len(station.departure_queue)
    normalized_time = env.now / SIMULATION_TIME
    
    # Calculate time until next event
    next_event = float('inf')
    if hasattr(env, 'next_arrival') and env.next_arrival < float('inf'):
        next_event = min(next_event, env.next_arrival - env.now)
    
    # Add information about trains that will soon depart
    soon_departing = 0
    for train in station.departure_queue:
        time_until_departure = max(0, train.dwell_time - (env.now - train.arrival_time - train.wait_time))
        if time_until_departure < 5:  # Departing in less than 5 time units
            soon_departing += 1
            next_event = min(next_event, time_until_departure)
    
    # Normalize next event time
    if next_event < float('inf'):
        normalized_next_event = min(1.0, next_event / 30)  # Cap at 30 time units
    else:
        normalized_next_event = 1.0
    
    return np.array([[
        waiting_count / 10,           # Normalized waiting count
        at_station_count / NUM_TRACKS, # Normalized utilization
        free_tracks / NUM_TRACKS,      # Normalized free tracks
        ready_to_depart / 10,          # Normalized ready to depart
        normalized_time               # Normalized simulation time
        # Removed normalized_next_event and soon_departing to match model input size
    ]])

def reward_function(station, action, env, prev_state, prev_throughput):
    reward = 0
    
    # Reward for throughput (trains processed)
    reward += (station.throughput - prev_throughput) * 10
    
    # Penalize long queues
    reward -= len(station.waiting_trains) * 0.5
    reward -= len(station.departure_queue) * 0.5
    
    # Reward for keeping tracks utilized but not overloaded
    utilization = station.tracks.count / NUM_TRACKS
    if 0.6 <= utilization <= 0.9:
        reward += 2  # Good utilization
    elif utilization > 0.9:
        reward -= 2  # Overutilization
    else:
        reward -= 1  # Underutilization
    
    # Small penalty for each time step to encourage efficiency
    reward -= 0.1
    
    return reward

def generate_charts():
    """Generate charts from simulation data and return as base64 strings"""
    try:
        with data_lock:
            throughput_history = simulation_data["throughput_history"]
            avg_wait_history = simulation_data["avg_wait_history"]
            exploration_history = simulation_data["exploration_history"]
            loss_history = simulation_data["loss_history"]
        
        # Create charts
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Throughput chart
        ax1.plot(throughput_history)
        ax1.set_title('Cumulative Throughput Over Time')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Trains Processed')
        ax1.grid(True)
        
        # Average wait time chart
        ax2.plot(avg_wait_history)
        ax2.set_title('Average Wait Time Over Time')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Average Wait Time (min)')
        ax2.grid(True)
        
        # Exploration rate chart
        ax3.plot(exploration_history)
        ax3.set_title('Exploration Rate Over Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Exploration Rate')
        ax3.grid(True)
        
        # Loss chart
        ax4.plot(loss_history)
        ax4.set_title('Training Loss Over Time')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Loss')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        
        # Encode as base64
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        return chart_base64
    except Exception as e:
        logger.error(f"Error generating charts: {e}")
        return None

def create_log_file():
    """Create a comprehensive log file with all simulation data"""
    try:
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_simulation_log_{timestamp}.txt"
        
        # Generate charts
        chart_base64 = generate_charts()
        
        # Create log content
        log_content = "TRAIN STATION SIMULATION - COMPREHENSIVE LOG\n"
        log_content += "=" * 60 + "\n\n"
        
        # Add simulation parameters
        log_content += "SIMULATION PARAMETERS:\n"
        log_content += f"Simulation Time: {SIMULATION_TIME}\n"
        log_content += f"Number of Tracks: {NUM_TRACKS}\n"
        log_content += f"Train Interval: {TRAIN_INTERVAL}\n"
        log_content += f"Gamma: {GAMMA}\n"
        log_content += f"Learning Rate: {LEARNING_RATE}\n"
        log_content += f"Exploration Range: {EXPLORATION_MIN} - {EXPLORATION_MAX}\n"
        log_content += f"Exploration Decay: {EXPLORATION_DECAY}\n\n"
        
        # Add summary statistics
        with data_lock:
            summary_stats = simulation_data["summary_stats"]
            train_events = simulation_data["train_events"]
        
        log_content += "SUMMARY STATISTICS:\n"
        for key, value in summary_stats.items():
            log_content += f"{key}: {value}\n"
        log_content += "\n"
        
        # Add train events
        log_content += "TRAIN EVENTS (last 100 events):\n"
        log_content += "Type\tTrain ID\tTime\tWait Time\tThroughput\n"
        for event in train_events[-100:]:
            log_content += f"{event['type']}\t{event['train_id']}\t{event['time']:.2f}\t{event['wait_time']:.2f}\t{event['throughput']}\n"
        log_content += "\n"
        
        # Add chart if available
        if chart_base64:
            log_content += "CHARTS:\n"
            log_content += f"<EMBEDDED_CHART:{chart_base64}>\n\n"
        
        # Add raw data
        log_content += "RAW DATA (first 10 entries of each):\n"
        with data_lock:
            for key in ["throughput_history", "avg_wait_history", "exploration_history", "loss_history"]:
                data = simulation_data[key][:10]
                log_content += f"{key}: {data}\n"
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(log_content)
        
        return filename
    except Exception as e:
        logger.error(f"Error creating log file: {e}")
        return None

def simulate(env, station, agent, train_counter):
    state = state_representation(station, env)
    total_reward = 0
    step = 0
    prev_throughput = station.throughput
    last_update_time = 0
    UPDATE_INTERVAL = 5  # Only update every 5 time units

    env.process(train_generator(env, station, train_counter))

    while env.now <= SIMULATION_TIME:
        # Get action from RL agent
        action = agent.act(state)
        
        # Process based on agent's decision
        reward = 0
        processed = False
        
        # Action 0: Process arrivals (if any waiting trains)
        if action == 0 and station.waiting_trains:
            # Try to allocate tracks to waiting trains
            free_tracks = NUM_TRACKS - station.tracks.count
            trains_to_process = min(free_tracks, len(station.waiting_trains))
            
            for i in range(trains_to_process):
                if station.waiting_trains:
                    train = station.waiting_trains.pop(0)
                    with station.tracks.request() as request:
                        yield request
                        # Train enters station
                        train.wait_time = env.now - train.arrival_time
                        station.update_metrics(train.wait_time)
                        
                        # Schedule departure after dwell time
                        yield env.timeout(train.dwell_time)
                        
                        # Train is ready to depart
                        station.departure_queue.append(train)
                        train.departure_time = env.now
                        
                        # Add to simulation data for broadcasting
                        with data_lock:
                            event = {
                                "type": "processed",
                                "train_id": train.id,
                                "time": env.now,
                                "wait_time": train.wait_time,
                                "throughput": station.throughput
                            }
                            simulation_data["train_events"].append(event)
                        
                        logger.info(f"Train {train.id} processed at time {env.now}, wait time: {train.wait_time}")
            
            processed = trains_to_process > 0
            reward = trains_to_process * 2  # Reward for processing trains
        
        # Action 1: Process departures (if any trains ready to depart)
        elif action == 1 and station.departure_queue:
            # Process all trains in departure queue
            for train in station.departure_queue[:]:
                station.departure_queue.remove(train)
                
                # Add to simulation data for broadcasting
                with data_lock:
                    event = {
                        "type": "departure",
                        "train_id": train.id,
                        "time": env.now,
                        "wait_time": train.wait_time,
                        "throughput": station.throughput
                    }
                    simulation_data["train_events"].append(event)
                
                logger.info(f"Train {train.id} departed at time {env.now}")
            
            processed = True
            reward = len(station.departure_queue) * 1  # Reward for clearing departures
        
        # If no action was taken, give a small penalty
        if not processed:
            reward = -0.5
        
        # Calculate time until next event
        next_event_time = float('inf')
        
        # Check when next train will arrive
        if hasattr(env, 'next_arrival') and env.next_arrival < float('inf'):
            next_event_time = min(next_event_time, env.next_arrival)
        
        # Check when next departure could happen (based on dwell times)
        for train in station.waiting_trains + [t for t in station.departure_queue if hasattr(t, 'arrival_time')]:
            if hasattr(train, 'arrival_time'):
                # Estimate when train could be processed
                potential_departure = train.arrival_time + train.dwell_time + 1  # +1 for processing time
                next_event_time = min(next_event_time, potential_departure)
        
        # If no events soon, advance to next likely event time
        if next_event_time == float('inf'):
            # No events scheduled, advance by a small amount
            time_advance = min(1, SIMULATION_TIME - env.now)
        else:
            time_advance = max(0.1, next_event_time - env.now)
        
        # Advance time
        if time_advance > 0:
            yield env.timeout(time_advance)
        
        # Get next state
        next_state = state_representation(station, env)
        
        # Calculate additional reward based on system state
        additional_reward = reward_function(station, action, env, state, prev_throughput)
        total_reward += reward + additional_reward
        prev_throughput = station.throughput
        
        # Remember experience
        done = env.now >= SIMULATION_TIME
        agent.remember(state, action, reward + additional_reward, next_state, done)
        
        # Train the agent
        loss = agent.replay()
        
        # Update state
        state = next_state
        step += 1
        
        # Update target network periodically
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        # Only update visualization periodically to avoid excessive updates
        if env.now - last_update_time >= UPDATE_INTERVAL:
            with data_lock:
                simulation_data["throughput_history"] = station.throughput_history.copy()
                simulation_data["avg_wait_history"] = station.avg_wait_history.copy()
                simulation_data["exploration_history"] = agent.exploration_history.copy()
                simulation_data["loss_history"] = agent.loss_history.copy()
            last_update_time = env.now

    # After simulation time, process any remaining trains
    logger.info("Processing remaining trains after simulation time...")
    
    # Process any trains still waiting
    while station.waiting_trains:
        free_tracks = NUM_TRACKS - station.tracks.count
        trains_to_process = min(free_tracks, len(station.waiting_trains))
        
        for i in range(trains_to_process):
            if station.waiting_trains:
                train = station.waiting_trains.pop(0)
                with station.tracks.request() as request:
                    yield request
                    train.wait_time = env.now - train.arrival_time
                    station.update_metrics(train.wait_time)
                    
                    # Schedule departure after dwell time
                    yield env.timeout(train.dwell_time)
                    
                    # Train is ready to depart
                    station.departure_queue.append(train)
                    train.departure_time = env.now
                    
                    logger.info(f"Train {train.id} processed after simulation, wait time: {train.wait_time}")
    
    # Process any trains ready to depart
    while station.departure_queue:
        train = station.departure_queue.pop(0)
        logger.info(f"Train {train.id} departed after simulation")
    
    logger.info(f"Simulation completed. Total reward: {total_reward}")
    logger.info(f"Total throughput: {station.throughput}")
    logger.info(f"Average wait time: {station.avg_wait_time} mins")
    logger.info(f"Trains arrived: {station.throughput + len(station.waiting_trains)}")
    logger.info(f"Trains departed: {station.throughput}")
    
    # Update simulation status and summary stats
    with data_lock:
        simulation_data["simulation_status"] = "completed"
        simulation_data["summary_stats"] = {
            "Total Throughput": station.throughput,
            "Average Wait Time": f"{station.avg_wait_time:.2f} mins",
            "Total Reward": f"{total_reward:.2f}",
            "Final Exploration Rate": f"{agent.exploration_rate:.4f}",
            "Simulation Duration": f"{SIMULATION_TIME} time units",
            "Trains Arrived": station.throughput + len(station.waiting_trains),
            "Trains Departed": station.throughput,
            "Trains Waiting": len(station.waiting_trains),
            "Completion Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    logger.info(f"Simulation completed. Total reward: {total_reward}")
    logger.info(f"Total throughput: {station.throughput}")
    logger.info(f"Average wait time: {station.avg_wait_time} mins")
    
    # Update simulation status and summary stats
    with data_lock:
        simulation_data["simulation_status"] = "completed"
        simulation_data["summary_stats"] = {
            "Total Throughput": station.throughput,
            "Average Wait Time": f"{station.avg_wait_time:.2f} mins",
            "Total Reward": f"{total_reward:.2f}",
            "Final Exploration Rate": f"{agent.exploration_rate:.4f}",
            "Simulation Duration": f"{SIMULATION_TIME} time units",
            "Completion Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def run_simulation():
    """Run the simulation in a separate thread"""
    global simulation_running
    try:
        with data_lock:
            simulation_data["simulation_status"] = "running"
        
        env = simpy.Environment()
        station = TrainStation(env, NUM_TRACKS)
        
        # Update state_size to match the simplified state representation (5 features)
        state_size = 5  # Matches the simplified state_representation output
        action_size = 2
        agent = DQNAgent(state_size, action_size)
        
        train_counter = iter(range(1, 1000))
        env.process(simulate(env, station, agent, train_counter))
        env.run(until=SIMULATION_TIME+50)
        
        # Save the trained model
        agent.save("train_traffic_dqn")
        logger.info("Training completed and model saved.")
        
        # Create log file
        log_filename = create_log_file()
        if log_filename:
            with data_lock:
                simulation_data["summary_stats"]["Log File"] = log_filename
            logger.info(f"Log file created: {log_filename}")
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
        with data_lock:
            simulation_data["simulation_status"] = "error"
    finally:
        simulation_running = False

# FastAPI setup
app = FastAPI(title="Train Station DQN Simulation")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total connections: {len(active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in active_connections:
            active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(active_connections)}")

manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Train Station DQN Simulation</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { display: flex; flex-direction: column; }
            .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .chart-container { position: relative; height: 300px; }
            .train-events { margin-top: 20px; }
            table { width: 100%; border-collapse: collapse; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .status { margin-bottom: 20px; padding: 10px; background-color: #f0f0f0; border-radius: 5px; }
            .controls { margin-bottom: 20px; display: flex; gap: 10px; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:disabled { background-color: #cccccc; cursor: not-allowed; }
            button:hover:not(:disabled) { background-color: #45a049; }
            #downloadButton { background-color: #2196F3; }
            #downloadButton:hover:not(:disabled) { background-color: #0b7dda; }
            .summary-stats { margin-top: 20px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; border-left: 4px solid #4CAF50; }
            .summary-stats h3 { margin-top: 0; }
            .stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
            .stat-item { padding: 5px; }
        </style>
    </head>
    <body>
        <h1>Train Station DQN Simulation</h1>
        <div class="controls">
            <button id="startButton" onclick="startSimulation()">Start Simulation</button>
            <button id="downloadButton" onclick="downloadLog()" disabled>Download Log</button>
        </div>
        <div class="status" id="status">Simulation not started. Click the button to begin.</div>
        
        <div class="summary-stats" id="summaryStats" style="display: none;">
            <h3>Simulation Summary</h3>
            <div class="stats-grid" id="statsGrid">
                <!-- Stats will be populated here -->
            </div>
        </div>
        
        <div class="container">
            <div class="charts">
                <div class="chart-container">
                    <canvas id="throughputChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="waitTimeChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="explorationChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
            </div>
            
            <div class="train-events">
                <h2>Train Events</h2>
                <table id="trainTable">
                    <thead>
                        <tr>
                            <th>Event</th>
                            <th>Train ID</th>
                            <th>Time</th>
                            <th>Wait Time</th>
                            <th>Throughput</th>
                        </tr>
                    </thead>
                    <tbody id="trainTableBody">
                    </tbody>
                </table>
            </div>
        </div>
        
        <script>
            // Initialize charts
            const throughputCtx = document.getElementById('throughputChart').getContext('2d');
            const waitTimeCtx = document.getElementById('waitTimeChart').getContext('2d');
            const explorationCtx = document.getElementById('explorationChart').getContext('2d');
            const lossCtx = document.getElementById('lossChart').getContext('2d');
            
            const throughputChart = new Chart(throughputCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Throughput',
                        data: [],
                        borderColor: 'blue',
                        borderWidth: 1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            const waitTimeChart = new Chart(waitTimeCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Wait Time',
                        data: [],
                        borderColor: 'red',
                        borderWidth: 1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            const explorationChart = new Chart(explorationCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Exploration Rate',
                        data: [],
                        borderColor: 'green',
                        borderWidth: 1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1
                        }
                    }
                }
            });
            
            const lossChart = new Chart(lossCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Training Loss',
                        data: [],
                        borderColor: 'purple',
                        borderWidth: 1,
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            
            // WebSocket connection
            let ws = null;
            
            function connectWebSocket() {
                // Get the current host and protocol
                const host = window.location.host;
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${host}/ws`;
                
                console.log("Connecting to WebSocket at:", wsUrl);
                ws = new WebSocket(wsUrl);
                
                ws.onopen = function() {
                    console.log("WebSocket connection established");
                    document.getElementById('status').textContent = 'Connected. Ready to start simulation.';
                };
                
                ws.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        console.log("Received data:", data);
                        
                        // Update charts
                        updateChart(throughputChart, data.throughput_history);
                        updateChart(waitTimeChart, data.avg_wait_history);
                        updateChart(explorationChart, data.exploration_history);
                        updateChart(lossChart, data.loss_history);
                        
                        // Update train events table
                        updateTrainTable(data.train_events);
                        
                        // Update status based on simulation state
                        if (data.simulation_status === 'running') {
                            document.getElementById('status').textContent = 'Simulation running...';
                            document.getElementById('startButton').disabled = true;
                        } else if (data.simulation_status === 'completed') {
                            document.getElementById('status').textContent = 'Simulation completed!';
                            document.getElementById('downloadButton').disabled = false;
                            
                            // Show summary statistics
                            updateSummaryStats(data.summary_stats);
                        } else if (data.simulation_status === 'error') {
                            document.getElementById('status').textContent = 'Simulation error!';
                            document.getElementById('startButton').disabled = false;
                        }
                    } catch (e) {
                        console.error("Error processing WebSocket message:", e);
                    }
                };
                
                ws.onerror = function(error) {
                    console.error("WebSocket error:", error);
                    document.getElementById('status').textContent = 'Connection error. Attempting to reconnect...';
                };
                
                ws.onclose = function() {
                    console.log("WebSocket connection closed");
                    document.getElementById('status').textContent = 'Connection closed. Attempting to reconnect...';
                    // Attempt to reconnect after a delay
                    setTimeout(connectWebSocket, 3000);
                };
            }
            
            function startSimulation() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({action: 'start_simulation'}));
                    document.getElementById('startButton').disabled = true;
                    document.getElementById('status').textContent = 'Starting simulation...';
                } else {
                    document.getElementById('status').textContent = 'WebSocket not connected. Cannot start simulation.';
                    // Try to reconnect
                    connectWebSocket();
                }
            }
            
            function downloadLog() {
                // This would typically request the log file from the server
                // For now, we'll just alert the user
                alert("Log file download functionality would be implemented here. The log file has been created on the server.");
                // In a real implementation, you would fetch the file from the server
                window.location.href = '/download-log';
            }
            
            function updateSummaryStats(stats) {
                const summaryDiv = document.getElementById('summaryStats');
                const statsGrid = document.getElementById('statsGrid');
                
                if (stats && Object.keys(stats).length > 0) {
                    summaryDiv.style.display = 'block';
                    statsGrid.innerHTML = '';
                    
                    for (const [key, value] of Object.entries(stats)) {
                        const statItem = document.createElement('div');
                        statItem.className = 'stat-item';
                        statItem.innerHTML = `<strong>${key}:</strong> ${value}`;
                        statsGrid.appendChild(statItem);
                    }
                }
            }
            
            function updateChart(chart, newData) {
                if (!newData || newData.length === 0) return;
                
                chart.data.labels = Array.from({length: newData.length}, (_, i) => i + 1);
                chart.data.datasets[0].data = newData;
                chart.update('none');
            }
            
            function updateTrainTable(events) {
                const tableBody = document.getElementById('trainTableBody');
                if (!events || events.length === 0) return;
                
                tableBody.innerHTML = '';
                
                // Reverse events to show latest first
                const reversedEvents = [...events].reverse();
                
                reversedEvents.forEach(event => {
                    const row = document.createElement('tr');
                    
                    const typeCell = document.createElement('td');
                    typeCell.textContent = event.type;
                    row.appendChild(typeCell);
                    
                    const idCell = document.createElement('td');
                    idCell.textContent = event.train_id;
                    row.appendChild(idCell);
                    
                    const timeCell = document.createElement('td');
                    timeCell.textContent = event.time.toFixed(2);
                    row.appendChild(timeCell);
                    
                    const waitCell = document.createElement('td');
                    waitCell.textContent = event.wait_time.toFixed(2);
                    row.appendChild(waitCell);
                    
                    const throughputCell = document.createElement('td');
                    throughputCell.textContent = event.throughput;
                    row.appendChild(throughputCell);
                    
                    tableBody.appendChild(row);
                });
            }
            
            // Initialize the connection
            connectWebSocket();
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle start simulation message
            if message.get('action') == 'start_simulation':
                global simulation_thread, simulation_running
                if not simulation_running:
                    simulation_running = True
                    simulation_thread = threading.Thread(target=run_simulation)
                    simulation_thread.daemon = True
                    simulation_thread.start()
                    logger.info("Simulation started via WebSocket command")
                else:
                    logger.info("Simulation already running")
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@app.get("/download-log")
async def download_log():
    """Endpoint to download the simulation log file"""
    with data_lock:
        if "Log File" in simulation_data["summary_stats"]:
            log_filename = simulation_data["summary_stats"]["Log File"]
            if os.path.exists(log_filename):
                return FileResponse(
                    path=log_filename,
                    filename=log_filename,
                    media_type="text/plain"
                )
    
    return {"error": "Log file not available yet"}

@app.on_event("startup")
async def startup_event():
    # Start the broadcast task (but not the simulation yet)
    asyncio.create_task(broadcast_data())
    logger.info("WebSocket broadcast started. Simulation will start on button press.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)