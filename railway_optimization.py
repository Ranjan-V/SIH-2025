import pandas as pd
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
import json
import re

# Load the CSV file
def load_train_data(csv_file='synthetic_indian_railways_detailed_3000.csv'):
    """
    Loads the synthetic train data from CSV.
    """
    df = pd.read_csv(csv_file)
    # Parse datetime columns if needed
    df['scheduled_departure'] = pd.to_datetime(df['scheduled_departure'])
    df['scheduled_arrival'] = pd.to_datetime(df['scheduled_arrival'])
    df['actual_departure'] = pd.to_datetime(df['actual_departure'])
    df['actual_arrival'] = pd.to_datetime(df['actual_arrival'])
    df['generated_timestamp'] = pd.to_datetime(df['generated_timestamp'])
    return df

def identify_delayed_trains(df):
    """
    Identifies delayed trains and their delay in minutes.
    """
    delayed_df = df[df['arr_delay_min'] > 0].copy()
    if not delayed_df.empty:
        print("Delayed Trains:")
        for _, row in delayed_df.iterrows():
            print(f"Train {row['train_number']} ({row['train_name']}): Delayed by {row['arr_delay_min']} minutes. Current location: {row['current_location']}")
        return delayed_df
    else:
        print("No delayed trains found.")
        return pd.DataFrame()

def find_common_section_blocks(df, station1='NDLS', station2='CNB', num_blocks=8):
    """
    Dynamically find block IDs involving two stations for the section.
    """
    section_blocks = []
    for _, row in df.iterrows():
        if pd.isna(row['route_block_ids']):
            continue
        blocks = row['route_block_ids'].split(';')
        for block in blocks:
            if station1 in block or station2 in block:
                section_blocks.append(block)
    unique_blocks = list(set(section_blocks))
    return [{'id': b} for b in unique_blocks[:num_blocks]]

def isTrainOnTrack(train, block_dict):
    """
    Determines if a train uses the block ID.
    """
    block_id = block_dict['id']
    if pd.isna(train['route_block_ids']):
        return False
    block_ids = train['route_block_ids'].split(';')
    return block_id in block_ids

def create_cp_sat_model_rolling_horizon(train_data, track_data, horizon_minutes=120):
    """
    Creates a CP-SAT model for train scheduling with a rolling horizon.
    Uses top 15 delayed trains for balanced optimization.
    """
    model = cp_model.CpModel()

    # Variables
    train_entry_times = {}
    train_exit_times = {}
    precedence = {}
    train_passes = {}

    HORIZON = horizon_minutes

    # Use top 15 delayed trains for optimization
    delayed_trains_df = train_data[train_data['arr_delay_min'] > 0]
    trains_on_section = delayed_trains_df.head(15).to_dict('records')
    print(f"Using {len(trains_on_section)} delayed trains for optimization.")

    if len(trains_on_section) == 0:
        print("No delayed trains for demo.")
        return None, None, None, None, None, None

    for train in trains_on_section:
        train_number = str(int(train['train_number']))

        train_entry_times[train_number] = model.NewIntVar(0, HORIZON, f'entry_{train_number}')
        train_exit_times[train_number] = model.NewIntVar(0, HORIZON, f'exit_{train_number}')
        train_passes[train_number] = model.NewBoolVar(f'passes_{train_number}')

        # Precedence only if same track
        for other_train in trains_on_section:
            other_train_number = str(int(other_train['train_number']))
            if train_number != other_train_number:
                is_same_track = any(
                    isTrainOnTrack(train, block) and isTrainOnTrack(other_train, block)
                    for block in track_data
                )
                if is_same_track:
                    key = (train_number, other_train_number)
                    if key not in precedence:
                        precedence[key] = model.NewBoolVar(f'prec_{train_number}_{other_train_number}')

    # Constraints
    # a. Minimum Travel Time (dynamic, capped at 90 min)
    for train in trains_on_section:
        train_number = str(int(train['train_number']))
        distance = train.get('distance_remaining_km', 100)
        speed = train.get('estimated_avg_speed_kmph', 60)
        min_travel_time = min(90, int((distance / max(speed, 1)) * 60)) if pd.notna(distance) and pd.notna(speed) else 45
        print(f"train_number, min_travel_time, HORIZON= {train_number}, {min_travel_time}, {HORIZON}")
        model.Add(train_entry_times[train_number] + min_travel_time <= train_exit_times[train_number])
        model.Add(train_exit_times[train_number] <= HORIZON).OnlyEnforceIf(train_passes[train_number])

    # b. Headway Constraint (10 min default)
    HEADWAY = 10
    for train_i in trains_on_section:
        train_num_i = str(int(train_i['train_number']))
        for train_j in trains_on_section:
            train_num_j = str(int(train_j['train_number']))
            if train_num_i != train_num_j:
                key_ij = (train_num_i, train_num_j)
                if key_ij in precedence:
                    key_ji = (train_num_j, train_num_i)
                    model.Add(train_entry_times[train_num_j] >= train_exit_times[train_num_i] + HEADWAY).OnlyEnforceIf(precedence[key_ij])
                    model.Add(train_entry_times[train_num_i] >= train_exit_times[train_num_j] + HEADWAY).OnlyEnforceIf(precedence[key_ji])
                    model.Add(precedence[key_ij] + precedence[key_ji] == 1)

    # c. Block Occupancy (no overlap)
    for block in track_data:
        trains_on_track = [train for train in trains_on_section if isTrainOnTrack(train, block)]
        for i in range(len(trains_on_track)):
            train_num_i = str(int(trains_on_track[i]['train_number']))
            for j in range(i + 1, len(trains_on_track)):
                train_num_j = str(int(trains_on_track[j]['train_number']))
                key_ij = (train_num_i, train_num_j)
                if key_ij in precedence:
                    model.Add(train_entry_times[train_num_j] >= train_exit_times[train_num_i]).OnlyEnforceIf(precedence[key_ij])
                    model.Add(train_entry_times[train_num_i] >= train_exit_times[train_num_j]).OnlyEnforceIf(precedence[(train_num_j, train_num_i)])

    # Objective: Maximize throughput
    objective = model.NewIntVar(0, len(trains_on_section), "total_trains")
    model.Add(objective == sum(train_passes.values()))
    model.Maximize(objective)

    return model, train_entry_times, train_exit_times, precedence, train_passes, trains_on_section

def solve_cp_sat_model(model):
    """
    Solves the CP-SAT model with extended timeout.
    """
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 180.0  # Increased to 3 minutes
    status = solver.Solve(model)

    print(f"Solver status: {solver.StatusName()}")
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution found!")
        print(f"Max Throughput: {solver.ObjectiveValue()}")
    else:
        print("No solution found.")
    return solver

def suggest_optimizations(solver, train_entry_times, train_exit_times, precedence, trains_on_section):
    """
    Suggests alternatives based on solver output.
    """
    print("\n=== Optimization Suggestions ===")
    print(f"Alternative: Re-sequencing on shared blocks to minimize delays, favoring express trains.")
    print(f"This maximizes section throughput by allowing {solver.ObjectiveValue()} trains in 120-min horizon vs. baseline.")
    
    for train in trains_on_section:
        train_number = str(int(train['train_number']))
        if train_number in train_entry_times and train_number in train_exit_times:
            print(f"Train {train_number}: Entry={solver.Value(train_entry_times[train_number])}, Exit={solver.Value(train_exit_times[train_number])}")

            for other_train in trains_on_section:
                if train_number != str(int(other_train['train_number'])):
                    other_train_number = str(int(other_train['train_number']))
                    key = (train_number, other_train_number)
                    if key in precedence:
                        if solver.Value(precedence[key]):
                            print(f"  Train {train_number} precedes Train {other_train_number}")
                        else:
                            print(f"  Train {other_train_number} precedes Train {train_number}")
            original_delay = train.get('arr_delay_min', 0)
            travel_time = solver.Value(train_exit_times[train_number]) - solver.Value(train_entry_times[train_number])
            print(f"  Estimated delay reduction: {max(0, original_delay - travel_time):.1f} min.")
        else:
            print(f"Train {train_number}: No entry/exit times found in solution.")

if __name__ == "__main__":
    df = load_train_data()
    print(f"Loaded {len(df)} trains.")

    identify_delayed_trains(df)

    # Get track_data for the section (increased to 8 blocks)
    track_data = find_common_section_blocks(df, num_blocks=8)
    print(f"Using {len(track_data)} blocks for section: {[b['id'] for b in track_data]}")

    horizon_minutes = 120
    model, train_entry_times, train_exit_times, precedence, train_passes, trains_on_section = create_cp_sat_model_rolling_horizon(
        df, track_data, horizon_minutes
    )
    if model is None:
        print("No model created.")
    else:
        solver = solve_cp_sat_model(model)
        if solver.StatusName() in ["OPTIMAL", "FEASIBLE"]:
            suggest_optimizations(solver, train_entry_times, train_exit_times, precedence, trains_on_section)
        else:
            print("Optimization not feasible. Try adjusting parameters.")