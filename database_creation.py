import pandas as pd
import random
import datetime

# Define lists for realism
station_codes = ['BPL', 'SUR', 'HYB', 'HWH', 'MDU', 'PURI', 'CBE', 'SBC', 'NDLS', 'JP', 'ASR', 'GHY', 'BCT', 'KYN', 'LKO', 'CNB', 'PRYJ', 'MAS', 'TVC', 'VSKP', 'NGP', 'MFP', 'BBS', 'NZM']
station_names = ['Bhopal Junction', 'Surat', 'Hyderabad', 'Howrah', 'Madurai', 'Puri', 'Coimbatore', 'Bengaluru', 'New Delhi', 'Jaipur', 'Amritsar', 'Guwahati', 'Mumbai CSMT', 'Kalyan', 'Lucknow', 'Kanpur Central', 'Prayagraj', 'Chennai Central', 'Trivandrum Central', 'Visakhapatnam', 'Nagpur', 'Mangalore', 'Bhubaneswar', 'Hazrat Nizamuddin']

train_types = ['Mail/Express', 'Tejas Express', 'Vande Bharat Express', 'Shatabdi Express', 'Rajdhani Express', 'Duronto Express', 'Superfast Express', 'Passenger', 'Freight', 'Goods']

priorities = [1,2,3,4,5]

statuses = ['Running', 'At Station', 'Upcoming', 'Completed', 'Cancelled']

loco_types = ['Electric', 'Diesel']

coach_comps = ["{'3A': 6, 'SL': 12}", "{'EC': 2, 'CC': 10}", "{'CC': 10, 'EC': 2}", "{'CC': 8, 'EC': 1}", "{'1A': 1, '2A': 4, '3A': 8}", "{'2A': 6, '3A': 10}", "{'SL': 8, '2S': 6}", "{'3A': 8, 'SL': 10, '2A': 2}"]

seats = [1296, 816, 624, 786, 1008, 1224, 1392]

# Function to generate random date time
def random_datetime(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + random.randint(0, delta.seconds)
    return start + datetime.timedelta(seconds=int_delta)

start_date = datetime.datetime(2025, 9, 1)
end_date = datetime.datetime(2025, 9, 30, 23, 59, 59)

data = []

for _ in range(3000):
    pnr = ''.join(random.choices('0123456789', k=10))
    train_number = random.randint(10000, 99999)
    train_type = random.choice(train_types)
    train_name = f"{train_type} {train_number}"
    priority = random.choice(priorities)
    origin_code = random.choice(station_codes)
    origin_name = next(name for code, name in zip(station_codes, station_names) if code == origin_code)
    dest_code = random.choice([c for c in station_codes if c != origin_code])
    dest_name = next(name for code, name in zip(station_codes, station_names) if code == dest_code)
    
    # Generate route: 4-12 stations
    num_stations = random.randint(4, 12)
    route_stations = [origin_code]
    available_stations = [c for c in station_codes if c != origin_code]
    while len(route_stations) < num_stations - 1:
        next_stat = random.choice(available_stations)
        route_stations.append(next_stat)
        available_stations.remove(next_stat)
    route_stations.append(dest_code)
    route_station_codes = '|'.join(route_stations)
    route_len_stations = len(route_stations)
    
    # Blocks
    num_blocks = len(route_stations) - 1
    block_ids = []
    block_dists = []
    for i in range(num_blocks):
        block_id = f"{route_stations[i]}_{route_stations[i+1]}_{random.randint(100, 9999)}"
        dist = random.randint(50, 300)
        block_ids.append(block_id)
        block_dists.append(dist)
    route_block_ids = ';'.join(block_ids)
    route_block_distances_km = ';'.join(map(str, block_dists))
    total_distance_km = sum(block_dists)
    
    # Times
    scheduled_departure = random_datetime(start_date, end_date).strftime('%Y-%m-%d %H:%M')
    travel_hours = total_distance_km / random.uniform(50, 100)
    scheduled_arrival_dt = datetime.datetime.strptime(scheduled_departure, '%Y-%m-%d %H:%M') + datetime.timedelta(hours=travel_hours)
    scheduled_arrival = scheduled_arrival_dt.strftime('%Y-%m-%d %H:%M')
    
    dep_delay = random.randint(-5, 30)
    actual_departure_dt = datetime.datetime.strptime(scheduled_departure, '%Y-%m-%d %H:%M') + datetime.timedelta(minutes=dep_delay)
    actual_departure = actual_departure_dt.strftime('%Y-%m-%d %H:%M')
    
    arr_delay = dep_delay + random.randint(-10, 40)
    actual_arrival_dt = scheduled_arrival_dt + datetime.timedelta(minutes=arr_delay)
    actual_arrival = actual_arrival_dt.strftime('%Y-%m-%d %H:%M')
    
    dep_delay_min = dep_delay
    arr_delay_min = arr_delay
    
    # Current location
    if random.random() < 0.3:
        current_location = random.choice(route_stations[:-1])
    else:
        block = random.choice(block_ids)
        current_location = f"between_{block}"
    distance_remaining_km = random.randint(0, total_distance_km)
    
    estimated_avg_speed_kmph = random.randint(60, 130)
    
    num_occupied = random.randint(1, min(4, num_blocks))
    occupied_blocks = '|'.join(random.sample(block_ids, num_occupied))
    
    origin_platforms = random.randint(1, 12)
    dest_platforms = random.randint(1, 12)
    origin_tracks = random.randint(2, 8)
    dest_tracks = random.randint(2, 8)
    origin_electrified = random.choice([True, False])
    dest_electrified = random.choice([True, False])
    loco_type = random.choice(loco_types)
    max_permissible_speed_kmph = random.randint(90, 160)
    headway_min = random.randint(3, 10)
    
    rake_id = f"RAKE_{random.randint(1000, 9999)}"
    crew_id = f"CRW_{random.randint(10000, 99999)}"
    maintenance_due = random.choice([True, False])
    reservation_load_pct = round(random.uniform(0, 100), 1)
    coach_composition = random.choice(coach_comps)
    estimated_total_seats = random.choice(seats)
    
    status = random.choice(statuses)
    canceled = random.choices([True, False], weights=[0.05, 0.95])[0]
    generated_timestamp = (datetime.datetime.now() - datetime.timedelta(days=random.randint(0,7))).strftime('%Y-%m-%d %H:%M:%S')
    
    row = {
        'pnr': pnr,
        'train_number': train_number,
        'train_name': train_name,
        'train_type': train_type,
        'priority': priority,
        'origin_code': origin_code,
        'origin_name': origin_name,
        'dest_code': dest_code,
        'dest_name': dest_name,
        'route_len_stations': route_len_stations,
        'route_station_codes': route_station_codes,
        'route_block_ids': route_block_ids,
        'route_block_distances_km': route_block_distances_km,
        'total_distance_km': total_distance_km,
        'scheduled_departure': scheduled_departure,
        'scheduled_arrival': scheduled_arrival,
        'actual_departure': actual_departure,
        'actual_arrival': actual_arrival,
        'dep_delay_min': dep_delay_min,
        'arr_delay_min': arr_delay_min,
        'current_location': current_location,
        'distance_remaining_km': distance_remaining_km,
        'estimated_avg_speed_kmph': estimated_avg_speed_kmph,
        'occupied_blocks': occupied_blocks,
        'origin_platforms': origin_platforms,
        'dest_platforms': dest_platforms,
        'origin_tracks': origin_tracks,
        'dest_tracks': dest_tracks,
        'origin_electrified': origin_electrified,
        'dest_electrified': dest_electrified,
        'loco_type': loco_type,
        'max_permissible_speed_kmph': max_permissible_speed_kmph,
        'headway_min': headway_min,
        'rake_id': rake_id,
        'crew_id': crew_id,
        'maintenance_due': maintenance_due,
        'reservation_load_pct': reservation_load_pct,
        'coach_composition': coach_composition,
        'estimated_total_seats': estimated_total_seats,
        'status': status,
        'canceled': canceled,
        'generated_timestamp': generated_timestamp
    }
    data.append(row)

df = pd.DataFrame(data)
df.to_csv('synthetic_indian_railways_detailed_3000.csv', index=False)
print("Dataset saved as 'synthetic_indian_railways_detailed_3000.csv'")