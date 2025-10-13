import simpy
import numpy as np
import random
import collections
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import json
import os

# Simulation parameters
SIMULATION_DISTANCE = 2182  # Total distance to cover in km
TRAIN_START_TIME = "16:10"  # Train starting time in 24-hour format

# Station parameters - dictionary with station_distance: dwell_time (minutes)
STATION_NAMES=['New Delhi', 'Mathura Junction', 'Billoch Pura', 'Agra Cantt', 'Dholpur Junction', 'Morena','Sank', 'Nurabad', 'Banmor', 'Rayaru', 'Birlanagar Jn','Gwalior Junction', 'VGL Jhansi Junction', 'Bina Junction','Ganj Basoda', 'Vidisha', 'Bhopal Junction', 'Rani Kamalapati', 'Narmadapuram', 'Itarsi Junction', 'Ghoradongri', 'Betul', 'Amla Junction', 'Pandhurna', 'Narkher Junction', 'Nagpur Junction', 'Sewagram Junction', 'Hinganghat', 'Chandrapur', 'Babupeth', 'Gondwana Visapur', 'Balharshah', 'Sirpur Kagaznagar', 'Bellampalli', 'Manchiryal', 'Ramagundam', 'Warangal', 'Khammam', 'Vijayawada Junction', 'Tenali Junction', 'Chirala', 'Ongole', 'Nellore',' Gudur Junction', 'MGR Chennai Central']

STATION_DWELL_TIMES = {
    # Example stations with varying distances and dwell times
    0: 0,    # Station at 0km - Initial Station
    141: 5,    # Station at 141km - medium station
    189: 2,    
    195: 5,   
    247: 2,   
    274: 2,   
    284: 0,   # Station at 284km - no stopage at this station
    287: 0,   
    293: 0,   
    300: 0,   
    310: 0,   
    313: 2,   
    410: 8,   # Station at 210km - major station
    563: 5,
    609: 2,
    649: 2,
    702: 5,
    708: 2,
    776: 2,
    794: 10,
    864: 1,
    901: 2,
    924: 2,
    988: 1,
    1006: 1,
    1092: 5,
    1169: 2,
    1202: 1,
    1287: 2,
    1301: 5,
    1371: 1,
    1409: 1,
    1429: 1,
    1443: 1,
    1544: 5,
    1652: 2,
    1751: 10,
    1782: 1,
    1840: 1,
    1889: 1,
    2006: 1,
    2004: 2
}

# Add more stations to reach 42 total stations
additional_stations = 42 - len(STATION_DWELL_TIMES)
if additional_stations > 0:
    # Generate additional stations with random distances and dwell times
    existing_distances = set(STATION_DWELL_TIMES.keys())
    min_distance = 5
    max_distance = SIMULATION_DISTANCE - 5
    
    for i in range(additional_stations):
        while True:
            distance = random.randint(min_distance, max_distance)
            # Ensure stations are at least 5km apart
            if all(abs(distance - existing_dist) >= 5 for existing_dist in existing_distances):
                break
        
        # Random dwell time between 2-8 minutes
        dwell_time = random.randint(2, 8)
        STATION_DWELL_TIMES[distance] = dwell_time
        existing_distances.add(distance)

# Sort stations by distance
STATION_DISTANCES = sorted(STATION_DWELL_TIMES.keys())

# Create a mapping from station distance to station name
def get_station_name(station_distance):
    """Get station name by finding the index of the station distance"""
    try:
        station_index = STATION_DISTANCES.index(station_distance)
        if station_index < len(STATION_NAMES):
            return STATION_NAMES[station_index]
        else:
            return f"Station {station_distance}km"  # Fallback for additional stations
    except ValueError:
        return f"Station {station_distance}km"  # Fallback if distance not found

print(f"Total stations: {len(STATION_DISTANCES)}")

# Improved DQN parameters
GAMMA = 0.95  # Reduced back for stability
LEARNING_RATE = 0.0003  # Further reduced learning rate
MEMORY_SIZE = 20000  # Increased memory size
BATCH_SIZE = 32  # Smaller batch size for stability
EXPLORATION_MAX = 0.7  # Reduced initial exploration
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995  # Slower decay for more gradual learning
TARGET_UPDATE_FREQ = 100  # Back to less frequent updates for stability

# Modified train parameters with realistic acceleration and jerk limits
MAX_SPEED = 120  # km/h (unchanged)
MIN_SPEED = 0     # km/h (changed from 20 to 0 to allow complete stops)

# Convert all rates to consistent units (per second)
TIME_STEP = 60  # 1 minute = 60 seconds

# Normal acceleration rate (km/h per second)
NORMAL_ACCELERATION_RATE = 30 / 60  # km/h per second (was 30 km/h per minute)

# Normal deceleration rate (km/h per second)
NORMAL_DECELERATION_RATE = 42 / 60  # km/h per second (was 42 km/h per minute)

# Emergency deceleration rate (km/h per second)
EMERGENCY_DECELERATION_RATE = 150 / 60  # km/h per second (was 150 km/h per minute)

# Jerk limits (km/h per second²)
COMFORTABLE_JERK_LIMIT = 65 / 3600  # km/h per second² (was 65 km/h per minute²)
MAX_ACCEPTABLE_JERK_LIMIT = 100 / 3600  # km/h per second² (was 100 km/h per minute²)

class EnvironmentConditions:
    def __init__(self):
        self.conditions = {
            'fog': 0.0,    # 0.0 to 1.0 (0 = no fog, 1 = heavy fog)
            'terrain': 0.0, # 0.0 to 1.0 (0 = flat, 1 = steep hills)
            'rain': 0.0,    # 0.0 to 1.0 (0 = no rain, 1 = heavy rain)
            'wind': 0.0,    # Added wind condition
            'visibility': 1.0  # Added visibility factor
        }
        
    def update(self, current_distance):
        # Simulate changing conditions based on distance
        # Fog is more likely in the morning (first 100km) with more realistic variation
        if current_distance < 100:
            self.conditions['fog'] = max(0, 0.7 - (current_distance / 100) + random.uniform(-0.1, 0.1))
        else:
            self.conditions['fog'] = random.uniform(0, 0.2) if random.random() < 0.05 else self.conditions['fog'] * 0.95
        
        # FIXED: Terrain variation with sine wave instead of monotonic increase
        terrain_base = 0.3  # Base terrain difficulty
        terrain_variation = 0.4 * np.sin(current_distance * 0.005)  # Sine wave variation
        terrain_noise = 0.1 * np.sin(current_distance * 0.02)  # Higher frequency noise
        self.conditions['terrain'] = max(0.0, min(1.0, terrain_base + terrain_variation + terrain_noise))
        
        # Random weather changes
        if random.random() < 0.08:  # 8% chance to change rain condition
            self.conditions['rain'] = random.uniform(0, 0.8)
        else:
            self.conditions['rain'] *= 0.98  # Gradually reduce rain
            
        # Wind conditions
        if random.random() < 0.05:  # 5% chance to change wind
            self.conditions['wind'] = random.uniform(0, 0.6)
        
        # Calculate visibility based on fog and rain
        self.conditions['visibility'] = max(0.1, 1.0 - (self.conditions['fog'] * 0.6 + self.conditions['rain'] * 0.3))
        
        return self.conditions

class ImprovedDQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.exploration_rate = EXPLORATION_MAX
        self.exploration_history = []
        self.loss_history = []
        self.reward_history = []
        self.cumulative_reward = 0
        self.cumulative_reward_history = []
        self.training_steps = 0

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Improved model architecture with dropout and better initialization"""
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu', 
                  kernel_initializer='he_normal'),
            Dropout(0.1),  # Light dropout to prevent overfitting
            Dense(128, activation='relu', kernel_initializer='he_normal'),
            Dropout(0.1),
            Dense(64, activation='relu', kernel_initializer='he_normal'),
            Dense(self.action_size, activation='linear')
        ])
        
        # Improved optimizer with gradient clipping and learning rate scheduling
        optimizer = Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
        model.compile(loss='huber', optimizer=optimizer)  # Huber loss for stability
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Improved epsilon-greedy strategy with warm-start bias
        if np.random.rand() < self.exploration_rate:
            # Bias towards acceleration in early stages when speed is low
            current_speed = state[0][0] * MAX_SPEED  # Denormalize speed
            if current_speed < 60 and np.random.rand() < 0.6:  # 60% chance to accelerate when speed < 60
                action = 2  # Accelerate
            else:
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

        # Double DQN improvement: use main network to select actions, target network to evaluate
        next_actions = np.argmax(self.model.predict(next_states, verbose=0), axis=1)

        for i, (state, action, reward, next_state, done) in enumerate(batch):
            if done:
                target_q = reward
            else:
                # Double DQN update rule with clipping to prevent overestimation
                target_q = reward + GAMMA * next_q[i][next_actions[i]]
                # Clip target to prevent extreme values and add smoothing
                target_q = np.clip(target_q, -50, 50)  # Reduced clipping range
            
            # Apply soft update instead of hard update for smoother learning
            current_q[i][action] = 0.9 * current_q[i][action] + 0.1 * target_q

        # Use Huber loss for more stable training
        history = self.model.fit(states, current_q, batch_size=BATCH_SIZE, verbose=0)
        loss = history.history['loss'][0]
        self.loss_history.append(loss)
        self.training_steps += 1

        # Improved exploration decay with minimum threshold
        if self.exploration_rate > EXPLORATION_MIN:
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate * EXPLORATION_DECAY)
        
        return loss

    def save(self, name):
        """Enhanced save function with metadata"""
        if not name.endswith('.weights.h5'):
            name += '.weights.h5'
        
        # Save model weights
        self.model.save_weights(name)
        
        # Save training metadata
        metadata = {
            'exploration_rate': self.exploration_rate,
            'training_steps': self.training_steps,
            'total_reward': self.cumulative_reward,
            'model_architecture': 'Improved DQN with dropout'
        }
        
        metadata_file = name.replace('.weights.h5', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model saved to {name}")
        print(f"Metadata saved to {metadata_file}")

    def load(self, name):
        if not name.endswith('.weights.h5'):
            name += '.weights.h5'
        
        if os.path.exists(name):
            self.model.load_weights(name)
            self.target_model.load_weights(name)
            print(f"Model loaded from {name}")
            
            # Try to load metadata
            metadata_file = name.replace('.weights.h5', '_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.exploration_rate = metadata.get('exploration_rate', EXPLORATION_MAX)
                    self.training_steps = metadata.get('training_steps', 0)
                    print(f"Metadata loaded: exploration_rate={self.exploration_rate:.3f}")
        else:
            print(f"Model file {name} not found")

class Train:
    def __init__(self, env):
        self.env = env
        self.speed = 0  # Start from stationary
        self.distance_covered = 0
        self.at_station = False
        self.station_dwell_time_remaining = 0
        self.current_station_index = 0
        self.station_dwell_times = STATION_DWELL_TIMES
        self.station_distances = STATION_DISTANCES
        self.next_station_distance = self.station_distances[0] if self.station_distances else SIMULATION_DISTANCE
        self.current_acceleration = 0  # Current acceleration rate (km/h per second)
        self.target_acceleration = 0   # Target acceleration rate (set by agent)
        self.last_action_time = 0      # Track when last action was taken
        self.energy_consumed = 0       # Track energy consumption
        self.comfort_score = 0         # Track passenger comfort
        self.arrival_time = None       # Track station arrival time
        self.departure_time = None     # Track station departure time
        self.start_time_minutes = self._parse_start_time(TRAIN_START_TIME)  # Starting time in minutes from midnight

    def _parse_start_time(self, time_str):
        """Convert time string (HH:MM) to minutes from midnight"""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    def _minutes_to_time_string(self, total_minutes):
        """Convert total minutes from start to 24-hour time format"""
        current_minutes = self.start_time_minutes + total_minutes
        # Handle day rollover
        current_minutes = current_minutes % (24 * 60)
        
        hours = int(current_minutes // 60)
        minutes = int(current_minutes % 60)
        return f"{hours:02d}:{minutes:02d}"

    def update_energy_consumption(self):
        """Calculate energy consumption based on speed and acceleration"""
        # Simple energy model: base consumption + acceleration penalty
        base_consumption = self.speed * 0.05  # Energy proportional to speed
        acceleration_penalty = abs(self.current_acceleration) * 2  # Penalty for acceleration
        self.energy_consumed += base_consumption + acceleration_penalty

class ImprovedVisualization:
    def __init__(self):
        plt.ion()
        self.fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Improved Train Speed Control RL Agent Performance', fontsize=16)

        # Flatten axes for easier access
        self.axes = axes.flatten()

        # Create all the plot lines
        self.distance_line, = self.axes[0].plot([], [], 'b-', label='Distance Covered', linewidth=2)
        self.station_markers = self.axes[0].scatter([], [], color='red', marker='^', s=50, label='Stations')
        
        self.speed_line, = self.axes[1].plot([], [], 'r-', label='Train Speed', linewidth=2)
        self.max_speed_line = self.axes[1].axhline(y=MAX_SPEED, color='r', linestyle='--', alpha=0.5, label='Max Speed')
        
        self.exploration_line, = self.axes[2].plot([], [], 'g-', label='Exploration Rate', linewidth=2)
        self.reward_line, = self.axes[3].plot([], [], 'm-', label='Cumulative Reward', linewidth=2)
        self.loss_line, = self.axes[4].plot([], [], 'orange', label='Training Loss', linewidth=2)
        
        # Environmental conditions plot
        self.fog_line, = self.axes[5].plot([], [], 'gray', label='Fog', alpha=0.7)
        self.rain_line, = self.axes[5].plot([], [], 'blue', label='Rain', alpha=0.7)
        self.terrain_line, = self.axes[5].plot([], [], 'brown', label='Terrain', alpha=0.7)
        self.visibility_line, = self.axes[5].plot([], [], 'green', label='Visibility', alpha=0.7)

        # Configure axes
        titles = ['Distance Covered Over Time', 'Train Speed', 'Exploration Rate', 
                 'Cumulative Reward', 'Training Loss', 'Environmental Conditions']
        xlabels = ['Time (minutes)', 'Time (minutes)', 'Time Step', 
                  'Time Step', 'Training Step', 'Time (minutes)']
        ylabels = ['Distance (km)', 'Speed (km/h)', 'Exploration Rate',
                  'Reward', 'Loss', 'Condition Value']

        for i, (ax, title, xlabel, ylabel) in enumerate(zip(self.axes, titles, xlabels, ylabels)):
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        # Set specific limits
        self.axes[0].set_ylim(0, SIMULATION_DISTANCE * 1.1)
        self.axes[1].set_ylim(0, MAX_SPEED * 1.1)
        self.axes[2].set_ylim(0, 1.0)
        self.axes[5].set_ylim(0, 1.0)

        plt.tight_layout()

    def update(self, time_history, distance_history, speed_history, exploration_history, 
               reward_history, loss_history, env_conditions_history):
        
        if not time_history:
            return
            
        # Update distance plot
        self.distance_line.set_data(time_history, distance_history)
        self.axes[0].relim()
        self.axes[0].autoscale_view()

        # Update station markers
        if distance_history and distance_history[-1] > 0:
            current_distance = distance_history[-1]
            current_time = time_history[-1]
            
            # Show stations that have been passed or are close
            relevant_stations = [dist for dist in STATION_DISTANCES if dist <= current_distance * 1.1]
            station_times = []
            station_dists = []
            
            for dist in relevant_stations:
                if dist <= current_distance:
                    # Estimate time when we passed this station
                    station_time = current_time * (dist / current_distance) if current_distance > 0 else 0
                    station_times.append(station_time)
                    station_dists.append(dist)
            
            if station_times:
                self.station_markers.set_offsets(np.column_stack([station_times, station_dists]))

        # Update speed plot
        self.speed_line.set_data(time_history, speed_history)
        self.axes[1].relim()
        self.axes[1].autoscale_view()

        # Update exploration rate
        if exploration_history:
            x_data = range(len(exploration_history))
            self.exploration_line.set_data(x_data, exploration_history)
            self.axes[2].relim()
            self.axes[2].autoscale_view()

        # Update cumulative reward
        if reward_history:
            x_data = range(len(reward_history))
            self.reward_line.set_data(x_data, reward_history)
            self.axes[3].relim()
            self.axes[3].autoscale_view()

        # Update loss
        if loss_history:
            x_data = range(len(loss_history))
            self.loss_line.set_data(x_data, loss_history)
            self.axes[4].relim()
            self.axes[4].autoscale_view()

        # Update environmental conditions
        if env_conditions_history and len(env_conditions_history) == len(time_history):
            fog_data = [conditions['fog'] for conditions in env_conditions_history]
            rain_data = [conditions['rain'] for conditions in env_conditions_history]
            terrain_data = [conditions['terrain'] for conditions in env_conditions_history]
            visibility_data = [conditions['visibility'] for conditions in env_conditions_history]
            
            self.fog_line.set_data(time_history, fog_data)
            self.rain_line.set_data(time_history, rain_data)
            self.terrain_line.set_data(time_history, terrain_data)
            self.visibility_line.set_data(time_history, visibility_data)
            self.axes[5].relim()
            self.axes[5].autoscale_view()

        plt.draw()
        plt.pause(0.01)

def improved_state_representation(train, env_conditions):
    # Enhanced state includes visibility and wind conditions
    # State includes:
    # 1. Current speed (normalized)
    # 2. Environmental conditions (fog, terrain, rain, wind, visibility)
    # 3. Distance covered so far (normalized)
    # 4. Remaining distance (normalized)
    # 5. At station flag
    # 6. Distance to next station (normalized)
    # 7. Current acceleration (normalized)
    # 8. Time of day factor (based on distance progress)
    
    normalized_speed = train.speed / MAX_SPEED
    normalized_distance = train.distance_covered / SIMULATION_DISTANCE
    remaining_distance = (SIMULATION_DISTANCE - train.distance_covered) / SIMULATION_DISTANCE
    at_station = 1.0 if train.at_station else 0.0
    
    # Calculate distance to next station (normalized by average station distance)
    if train.current_station_index < len(train.station_distances):
        distance_to_next_station = (train.next_station_distance - train.distance_covered) / 50  # Normalize by average distance
    else:
        distance_to_next_station = (SIMULATION_DISTANCE - train.distance_covered) / 50
    
    # Normalize acceleration relative to maximum possible
    max_possible_accel = max(NORMAL_ACCELERATION_RATE, abs(NORMAL_DECELERATION_RATE))
    normalized_acceleration = train.current_acceleration / max_possible_accel
    
    # Time factor (cyclical, representing time of day effect)
    time_factor = np.sin(2 * np.pi * normalized_distance)
    
    return np.array([[
        normalized_speed,
        env_conditions['fog'],
        env_conditions['terrain'],
        env_conditions['rain'],
        env_conditions['wind'],
        env_conditions['visibility'],
        normalized_distance,
        remaining_distance,
        at_station,
        distance_to_next_station,
        normalized_acceleration,
        time_factor
    ]])

def improved_reward_function(train, action, env_conditions, prev_distance_covered, current_time, prev_speed=0):
    # Enhanced reward function with better balance and early speed incentive
    progress_reward = (train.distance_covered - prev_distance_covered) * 12
    
    # Early speed boost incentive - encourage higher speeds in first 500km
    speed_incentive = 0
    if train.distance_covered < 500:
        if train.speed >= 80:
            speed_incentive = 3.0  # Strong reward for maintaining good speed early
        elif train.speed >= 60:
            speed_incentive = 1.5
        elif train.speed < 40:
            speed_incentive = -2.0  # Penalty for being too slow early
    
    # Enhanced safety penalty considering visibility
    safety_penalty = 0
    safe_speed_threshold = MAX_SPEED * env_conditions['visibility'] * 0.9
    if train.speed > safe_speed_threshold:
        safety_penalty = -4 * (train.speed - safe_speed_threshold) / MAX_SPEED
    
    # Efficiency penalty/reward - adjusted for conditions
    efficiency_reward = 0
    if not train.at_station:
        if env_conditions['visibility'] > 0.7:  # Good visibility
            if 75 <= train.speed <= 105:
                efficiency_reward = 2.5
            elif train.speed < 50:
                efficiency_reward = -2.0
        else:  # Poor visibility
            if 40 <= train.speed <= 80:
                efficiency_reward = 2.0
            elif train.speed > 90:
                efficiency_reward = -1.5

    # Comfort penalty for harsh speed changes
    comfort_penalty = 0
    speed_change = abs(train.speed - prev_speed)
    if speed_change > COMFORTABLE_JERK_LIMIT * TIME_STEP * 30:  # Threshold for comfort
        comfort_penalty = -1.5 * (speed_change / MAX_SPEED)
        train.comfort_score -= 1

    # Energy efficiency penalty
    energy_penalty = 0
    if train.current_acceleration > NORMAL_ACCELERATION_RATE * 0.8 and train.speed > 90:
        energy_penalty = -0.8
    
    train.update_energy_consumption()

    # Station arrival bonus
    station_bonus = 0
    if train.at_station and train.station_dwell_time_remaining == train.station_dwell_times.get(train.next_station_distance, 5):
        station_bonus = 12

    # Small time penalty to encourage completion
    time_penalty = -0.03

    # Weather adaptation bonus
    weather_bonus = 0
    if (env_conditions['rain'] > 0.5 or env_conditions['fog'] > 0.4) and train.speed <= 80:
        weather_bonus = 1.0
    elif env_conditions['visibility'] > 0.9 and 80 <= train.speed <= 110:
        weather_bonus = 0.5

    total_reward = (progress_reward + speed_incentive + safety_penalty + efficiency_reward + 
                   comfort_penalty + energy_penalty + station_bonus + 
                   time_penalty + weather_bonus)
    
    return total_reward

def simulate(env, agent, env_conditions, visualization=None):
    # Create a train
    train = Train(env)
    
    state = improved_state_representation(train, env_conditions.conditions)
    total_reward = 0
    step = 0
    prev_distance_covered = 0
    prev_speed = 0
    
    # History for visualization
    time_history = []
    distance_history = []
    speed_history = []
    env_conditions_history = []
    
    # MODIFIED: Simplified print statement
    print("Starting simulation with improved DQN agent...")
    
    while train.distance_covered < SIMULATION_DISTANCE:
        # Update environmental conditions based on current distance
        current_conditions = env_conditions.update(train.distance_covered)
        env_conditions_history.append(current_conditions.copy())
        
        # Check if we've reached a station
        if (train.current_station_index < len(train.station_distances) and 
            train.distance_covered >= train.next_station_distance and 
            not train.at_station):
            
            train.at_station = True
            train.arrival_time = env.now
            # Get dwell time from dictionary using station distance as key
            dwell_time = train.station_dwell_times.get(train.next_station_distance, 5)
            train.station_dwell_time_remaining = dwell_time
            train.speed = 0  # Stop at station
            train.current_acceleration = 0
            train.target_acceleration = 0
            # MODIFIED: Print station name and arrival time in 24-hour format and dwell time in minutes
            station_name = get_station_name(train.next_station_distance)
            arrival_time_str = train._minutes_to_time_string(train.arrival_time)
            print(f"{station_name}: Arrival {arrival_time_str}, Dwell {dwell_time}min")
        
        # Handle station dwell time
        if train.at_station:
            # Decrement dwell time each minute
            train.station_dwell_time_remaining -= 1
            
            # Depart only when dwell time is fully completed (goes negative)
            if train.station_dwell_time_remaining < 0:
                train.at_station = False
                train.departure_time = env.now
                # MODIFIED: Print station name and departure time in 24-hour format
                station_name = get_station_name(train.next_station_distance)
                departure_time_str = train._minutes_to_time_string(train.departure_time)
                print(f"{station_name}: Departure {departure_time_str}")
                
                train.current_station_index += 1
                # Set next station distance
                if train.current_station_index < len(train.station_distances):
                    train.next_station_distance = train.station_distances[train.current_station_index]
                else:
                    train.next_station_distance = SIMULATION_DISTANCE  # Final destination
                # Reset acceleration to encourage smooth acceleration after station
                train.current_acceleration = NORMAL_ACCELERATION_RATE * 1.2
                train.target_acceleration = NORMAL_ACCELERATION_RATE * 1.2
        
        # Only get action from agent if not at station
        action = 1  # Default action (maintain speed)
        if not train.at_station:
            action = agent.act(state)
            train.last_action_time = env.now
            
            # Set target acceleration based on action
            if action == 0:  # Decrease speed
                train.target_acceleration = -NORMAL_DECELERATION_RATE
            elif action == 2:  # Increase speed
                train.target_acceleration = NORMAL_ACCELERATION_RATE
            else:  # Maintain speed
                train.target_acceleration = 0
        
        if not train.at_station:
            # Apply jerk limits to smoothly change acceleration
            acceleration_diff = train.target_acceleration - train.current_acceleration
            max_acceleration_change = COMFORTABLE_JERK_LIMIT * TIME_STEP
            
            if abs(acceleration_diff) > max_acceleration_change:
                if acceleration_diff > 0:
                    train.current_acceleration += max_acceleration_change
                else:
                    train.current_acceleration -= max_acceleration_change
            else:
                train.current_acceleration = train.target_acceleration
            
            # Update speed using proper physics: v = v₀ + a*t
            prev_speed = train.speed
            new_speed = train.speed + (train.current_acceleration * TIME_STEP)  # km/h
            
            # Apply environmental speed limits based on visibility and conditions
            max_safe_speed = MAX_SPEED
            max_safe_speed *= current_conditions['visibility']  # Reduce speed in low visibility
            max_safe_speed *= (1 - current_conditions['terrain'] * 0.15)  # Reduce speed on difficult terrain
            max_safe_speed *= (1 - current_conditions['rain'] * 0.2)  # Reduce speed in rain
            max_safe_speed *= (1 - current_conditions['wind'] * 0.1)  # Reduce speed in high wind
            
            new_speed = max(MIN_SPEED, min(new_speed, max_safe_speed))
            
            # Enhanced station approach - more gradual and intelligent slowing down
            if (train.current_station_index < len(train.station_distances) and
                (train.next_station_distance - train.distance_covered) < 15):  # Start slowing down 15km before station
                
                # Calculate distance to station
                distance_to_station = train.next_station_distance - train.distance_covered
                
                # More sophisticated approach speed calculation
                if distance_to_station > 10:
                    target_station_approach_speed = max(20, (distance_to_station / 15) * 60)
                elif distance_to_station > 5:
                    target_station_approach_speed = max(10, (distance_to_station / 10) * 40)
                else:
                    target_station_approach_speed = max(5, (distance_to_station / 5) * 20)
                
                if new_speed > target_station_approach_speed:
                    # Calculate required deceleration for smooth approach
                    required_deceleration = (new_speed - target_station_approach_speed) / (TIME_STEP * 2)
                    train.target_acceleration = -min(required_deceleration, NORMAL_DECELERATION_RATE)
            
            train.speed = new_speed
            
            # Calculate distance covered using average speed over the time step
            average_speed = (prev_speed + train.speed) / 2
            distance_covered_this_step = (average_speed * TIME_STEP) / 3600  # km
            train.distance_covered += distance_covered_this_step
        
        # Calculate reward with improved function
        next_state = improved_state_representation(train, current_conditions)
        reward = improved_reward_function(train, action, current_conditions, prev_distance_covered, env.now, prev_speed)
        
        # Additional rewards for good behavior
        if not train.at_station:
            # Bonus for maintaining optimal speed in good conditions
            if current_conditions['visibility'] > 0.8 and 80 <= train.speed <= 110:
                reward += 1.5
            elif train.speed > 120:
                reward -= 1.0  # Penalty for excessive speed
            
            # Bonus for smooth acceleration/deceleration
            if abs(train.current_acceleration) < COMFORTABLE_JERK_LIMIT * TIME_STEP * 0.5:
                reward += 0.8
        
        prev_distance_covered = train.distance_covered
        total_reward += reward
        agent.reward_history.append(reward)
        agent.cumulative_reward += reward
        agent.cumulative_reward_history.append(agent.cumulative_reward)
        
        done = train.distance_covered >= SIMULATION_DISTANCE
        
        # Remember state for training
        if not train.at_station:
            agent.remember(state, action, reward, next_state, done)
        
        # Train the agent with improved frequency and stability checks
        if not done and not train.at_station and len(agent.memory) > BATCH_SIZE and step % 5 == 0:  # Less frequent training
            loss = agent.replay()
            # Early stopping if loss becomes too high
            if loss > 1000:  # Higher threshold for unstable training
                print(f"Warning: Very high loss detected ({loss:.2f}). Training may be unstable.")
        
        state = next_state
        step += 1
        
        # Update visualization
        time_history.append(env.now)
        distance_history.append(train.distance_covered)
        speed_history.append(train.speed)
        
        if step % TARGET_UPDATE_FREQ == 0:
            agent.update_target_model()
        
        if visualization and step % 10 == 0:  # Update visualization every 10 steps for better performance
            visualization.update(
                time_history, distance_history, speed_history,
                agent.exploration_history, agent.cumulative_reward_history,
                agent.loss_history, env_conditions_history
            )
        
        yield env.timeout(1)  # Each step is 1 minute of simulation time
    
    # MODIFIED: Updated final summary to show actual journey times
    print(f"\n=== Simulation Completed ===")
    start_time_str = train._minutes_to_time_string(0)
    end_time_str = train._minutes_to_time_string(env.now)
    print(f"Journey: {start_time_str} to {end_time_str} (Duration: {env.now:.1f} minutes / {env.now/60:.1f} hours)")
    print(f"Total distance: {train.distance_covered:.2f} km")
    print(f"Average speed: {(train.distance_covered / env.now * 60):.2f} km/h")
    
    return env.now

def train_agent():
    env = simpy.Environment()
    env_conditions = EnvironmentConditions()
    state_size = 12  # Updated state size for improved representation
    action_size = 3  # decrease speed, maintain speed, increase speed
    agent = ImprovedDQNAgent(state_size, action_size)
    visualization = ImprovedVisualization()
    
    # MODIFIED: Simplified initial print
    print("=== Training Improved DQN Agent ===")
    
    # Run the simulation
    sim_process = simulate(env, agent, env_conditions, visualization)
    env.process(sim_process)
    
    # Estimate max time needed and run the simulation
    max_time = SIMULATION_DISTANCE * 60 / 20 * 3  # Increased buffer for training
    
    try:
        env.run(until=max_time)
    except Exception as e:
        print(f"Simulation ended with exception: {e}")
    
    plt.ioff()
    plt.show()
    return agent, env.now

def plot_final_results(agent, total_time):
    """Enhanced results plotting with more detailed analysis"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Improved DQN Training Results Summary', fontsize=16)
    
    axes = axes.flatten()

    # Plot 1: Exploration Rate
    if agent.exploration_history:
        axes[0].plot(agent.exploration_history, 'g-', linewidth=2)
        axes[0].set_title('Exploration Rate Decay')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Exploration Rate')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.0)

    # Plot 2: Training Loss (with smoothing)
    if agent.loss_history:
        # Apply moving average for smoother visualization
        window_size = min(50, len(agent.loss_history) // 10)
        if window_size > 1:
            smoothed_loss = np.convolve(agent.loss_history, np.ones(window_size)/window_size, mode='valid')
            axes[1].plot(smoothed_loss, 'orange', linewidth=2, label='Smoothed Loss')
            axes[1].plot(agent.loss_history, 'orange', alpha=0.3, linewidth=1, label='Raw Loss')
            axes[1].legend()
        else:
            axes[1].plot(agent.loss_history, 'orange', linewidth=2)
        axes[1].set_title('Training Loss')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].grid(True, alpha=0.3)

    # Plot 3: Reward per Step
    if agent.reward_history:
        # Show both raw and smoothed rewards
        window_size = min(100, len(agent.reward_history) // 20)
        if window_size > 1:
            smoothed_rewards = np.convolve(agent.reward_history, np.ones(window_size)/window_size, mode='valid')
            axes[2].plot(smoothed_rewards, 'm-', linewidth=2, label='Smoothed Reward')
            axes[2].plot(agent.reward_history, 'm-', alpha=0.3, linewidth=1, label='Raw Reward')
            axes[2].legend()
        else:
            axes[2].plot(agent.reward_history, 'm-', linewidth=2)
        axes[2].set_title('Reward per Step')
        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Reward')
        axes[2].grid(True, alpha=0.3)

    # Plot 4: Cumulative Reward
    if agent.cumulative_reward_history:
        axes[3].plot(agent.cumulative_reward_history, 'b-', linewidth=2)
        axes[3].set_title('Cumulative Reward')
        axes[3].set_xlabel('Time Step')
        axes[3].set_ylabel('Cumulative Reward')
        axes[3].grid(True, alpha=0.3)

    # Plot 5: Training Statistics
    axes[4].text(0.1, 0.8, f'Final Exploration Rate: {agent.exploration_rate:.4f}', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].text(0.1, 0.7, f'Training Steps: {agent.training_steps}', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].text(0.1, 0.6, f'Total Reward: {agent.cumulative_reward:.2f}', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].text(0.1, 0.5, f'Simulation Time: {total_time:.1f} minutes', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].text(0.1, 0.4, f'Average Speed: {(SIMULATION_DISTANCE / total_time * 60):.2f} km/h', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].text(0.1, 0.3, f'Memory Size: {len(agent.memory)}', 
                transform=axes[4].transAxes, fontsize=12)
    axes[4].set_title('Training Statistics')
    axes[4].set_xlim(0, 1)
    axes[4].set_ylim(0, 1)
    axes[4].axis('off')

    # Plot 6: Performance Metrics
    if agent.reward_history:
        # Calculate performance metrics
        recent_rewards = agent.reward_history[-100:] if len(agent.reward_history) > 100 else agent.reward_history
        avg_recent_reward = np.mean(recent_rewards)
        
        early_rewards = agent.reward_history[:100] if len(agent.reward_history) > 100 else agent.reward_history[:len(agent.reward_history)//2]
        avg_early_reward = np.mean(early_rewards) if early_rewards else 0
        
        improvement = avg_recent_reward - avg_early_reward
        
        axes[5].bar(['Early Training', 'Recent Training'], [avg_early_reward, avg_recent_reward], 
                   color=['red', 'green'], alpha=0.7)
        axes[5].set_title(f'Performance Improvement\n(+{improvement:.2f} reward)')
        axes[5].set_ylabel('Average Reward')
        axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('improved_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # MODIFIED: Enhanced initial display with starting time information
    print("=== Enhanced Train Speed Control DQN Simulation ===")
    print(f"Train Start Time: {TRAIN_START_TIME}")
    print(f"Total stations: {len(STATION_DISTANCES)}")
    print(f"Total distance: {SIMULATION_DISTANCE} km")
    print(f"Maximum speed: {MAX_SPEED} km/h")
    
    # Train the improved agent
    print(f"\nStarting training...")
    trained_agent, total_time = train_agent()
    
    # Save the trained model
    trained_agent.save("improved_train_speed_control_dqn")
    
    # Plot final results
    plot_final_results(trained_agent, total_time)
    
