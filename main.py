# __main__ 블록에 있는 run_simulation_parallel, run_simulation 으로 각 병렬과 직렬 테스트 가능

# 원본 코드와 비교한 주요 변경사항:

# 1. 객체 지향적 설계
#    - Person, House, Building 클래스 추가
#    - 각 개체를 객체로 모델링하여 관리

# 2. 더 복잡한 모델링 기능
#    - 지도 생성 함수 (generate_map)
#    - 사람 생성 함수 (generate_people)
#    - 경로 찾기 알고리즘 (find_path, find_nearest_road)

# 3. 감염 모델의 세분화
#    - 마스크 착용 상태에 따른 감염률 차별화
#    - 연령에 따른 회복률과 사망률 차별화
#    - 실내와 실외의 감염 확산 모델 분리

# 4. 병렬 처리 기능
#    - MPI를 활용한 영역 분할 병렬 처리
#    - 프로세스 간 사람 데이터 교환 기능
#    - 분산된 데이터 수집 기능

# 5. 시뮬레이션 흐름 구조화
#    - 하루 단위의 시뮬레이션 사이클
#    - 출근-업무-귀가 패턴 구현
#    - 일과 중 감염 시뮬레이션과 집에서의 감염 시뮬레이션 분리

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import time
import sys
from mpi4py import MPI
from timing_results import TimingManager

# Map Legend:
# 0: empty space
# 1: road
# 200-299: house with ID (e.g., 200 is house ID 0, 201 is house ID 1)
# 300-399: building with ID (e.g., 300 is building ID 0, 301 is building ID 1)

# Person Class to track individuals separately from the map
class Person:
    def __init__(self, person_id, x, y, house_id, age_group, mask_status, infection_status, dest_type=None, dest_id=None):
        self.id = person_id
        self.x = x  # global x coordinate on the map
        self.y = y  # global y coordinate on the map
        self.house_id = house_id  # house this person belongs to
        self.age_group = age_group  # 0: child, 1: adult, 2: senior
        self.mask_status = mask_status  # 0: no mask, 1: mask, 2: mask+face shield
        self.infection_status = infection_status  # 0: normal, 1: infected, 2: recovered, 3: dead
        self.dest_type = dest_type  # 'house' or 'building' or None if not moving
        self.dest_id = dest_id  # ID of the destination 
    
    def __str__(self):
        status = ["normal", "infected", "recovered", "dead"][self.infection_status]
        mask = ["no mask", "mask", "mask+shield"][self.mask_status]
        age = ["child", "adult", "senior"][self.age_group]
        return f"Person {self.id} at ({self.x},{self.y}), House: {self.house_id}, Age: {age}, Mask: {mask}, Status: {status}"

# House and Building classes to track their information
class House:
    def __init__(self, house_id, top_left_x, top_left_y, size):
        self.id = house_id
        self.x = top_left_x
        self.y = top_left_y
        self.size = size
        
    def contains_point(self, x, y):
        return (self.x <= x < self.x + self.size) and (self.y <= y < self.y + self.size)
        
class Building:
    def __init__(self, building_id, top_left_x, top_left_y, size):
        self.id = building_id
        self.x = top_left_x
        self.y = top_left_y
        self.size = size
        
    def contains_point(self, x, y):
        return (self.x <= x < self.x + self.size) and (self.y <= y < self.y + self.size)

def generate_map(size=100, houses=5, house_size=3, buildings=3, building_size=5, road_width=6, roads=1):
    """
    Generate a map with houses, buildings, and roads.
    
    Parameters:
    - size: Size of the map (size x size).
    - houses: Number of houses to place.
    - house_size: Size of each house (house_size x house_size).
    - buildings: Number of buildings to place.
    - building_size: Size of each building (building_size x building_size).
    - road_width: Width of the roads.
    
    Returns:
    - map_array: 2D numpy array representing the map with buildings and houses
    - houses_list: List of House objects with their locations
    - buildings_list: List of Building objects with their locations
    """

    # 지도 생성
    map_array = np.zeros((size, size), dtype=int)

    # Create the specified number of roads at random positions
    # Minimum distance between roads is 6 cells
    
    # Lists to track road positions
    vertical_positions = []
    horizontal_positions = []
    
    # Generate vertical roads
    for _ in range(roads):
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            # Random position for vertical road
            pos = random.randint(road_width, size - road_width)
            
            # Check if it's at least 6 cells away from other roads
            if all(abs(pos - existing) >= 6 for existing in vertical_positions):
                vertical_positions.append(pos)
                map_array[:, pos:pos + road_width] = 1  # Create vertical road
                break
            attempts += 1
    
    # Generate horizontal roads
    for _ in range(roads):
        attempts = 0
        while attempts < 100:  # Limit attempts to avoid infinite loop
            # Random position for horizontal road
            pos = random.randint(road_width, size - road_width)
            
            # Check if it's at least 6 cells away from other roads
            if all(abs(pos - existing) >= 6 for existing in horizontal_positions):
                horizontal_positions.append(pos)
                map_array[pos:pos + road_width, :] = 1  # Create horizontal road
                break
            attempts += 1
    
    # **도로에 맞닿아 있도록** 집과 건물 배치
    houses_list = []
    buildings_list = []

    # 집 배치
    for i in range(houses):
        while True:
            x = random.randint(0, size - house_size)
            y = random.randint(0, size - house_size)
            if np.all(map_array[x:x + house_size, y:y + house_size] == 0):
                # 집이 도로와 맞닿도록
                if ((y > 0 and np.any(map_array[x:x + house_size, y - 1] == 1)) or
                    (y + house_size < size and np.any(map_array[x:x + house_size, y + house_size] == 1)) or
                    (x > 0 and np.any(map_array[x - 1, y:y + house_size] == 1)) or
                    (x + house_size < size and np.any(map_array[x + house_size, y:y + house_size] == 1))):
                    map_array[x:x + house_size, y:y + house_size] = 200 + i  # 집 번호는 200부터 시작
                    houses_list.append(House(i, x, y, house_size))
                    break
    
    # 건물 배치
    for i in range(buildings):
        while True:
            x = random.randint(0, size - building_size)
            y = random.randint(0, size - building_size)
            if np.all(map_array[x:x + building_size, y:y + building_size] == 0):
                # 건물이 도로와 맞닿도록
                if ((y > 0 and np.any(map_array[x:x + building_size, y - 1] == 1)) or
                    (y + building_size < size and np.any(map_array[x:x + building_size, y + building_size] == 1)) or
                    (x > 0 and np.any(map_array[x - 1, y:y + building_size] == 1)) or
                    (x + building_size < size and np.any(map_array[x + building_size, y:y + building_size] == 1))):
                    map_array[x:x + building_size, y:y + building_size] = 300 + i  # 건물 번호는 300부터 시작
                    buildings_list.append(Building(i, x, y, building_size))
                    break

    return map_array, houses_list, buildings_list

def generate_people(map_array, houses_list, person_min_per_house=10, person_max_per_house=20, 
                 age_distribution=[0.2, 0.6, 0.2], infection_rate=0.1, mask_distribution=[0.5, 0.3, 0.2]):
    """
    Generate people and place them in houses.
    
    Parameters:
    - map_array: The map array
    - houses_list: List of House objects
    - person_min_per_house: Minimum number of people per house
    - person_max_per_house: Maximum number of people per house
    - age_distribution: Distribution of age groups [child, adult, senior]
    - infection_rate: Initial infection rate
    - mask_distribution: Distribution of mask usage [none, mask, mask+shield]
    
    Returns:
    - people_list: List of Person objects
    """
    # check if the age distribution is valid
    if len(age_distribution) != 3 or not np.isclose(sum(age_distribution), 1.0):
        raise ValueError("Age distribution must be a list of three probabilities that sum to 1.")
    
    # check if the mask distribution is valid
    if len(mask_distribution) != 3 or not np.isclose(sum(mask_distribution), 1.0):
        raise ValueError("Mask distribution must be a list of three probabilities that sum to 1.")
        
    people_list = []
    person_id_counter = 1
    
    for house in houses_list:
        num_people = random.randint(person_min_per_house, person_max_per_house)
        house_x, house_y = house.x, house.y
        house_size = house.size
        
        for _ in range(num_people):
            age_group = np.random.choice([0, 1, 2], p=age_distribution)
            mask_status = np.random.choice([0, 1, 2], p=mask_distribution)
            infection_status = 0 if random.random() > infection_rate else 1
            
            # Find an empty spot in the house
            while True:
                # Local coordinates within the house
                local_x = random.randint(0, house_size - 1)
                local_y = random.randint(0, house_size - 1)
                
                # Convert to global coordinates
                global_x = house_x + local_x
                global_y = house_y + local_y
                
                # Create a new person
                person = Person(
                    person_id=person_id_counter,
                    x=global_x, 
                    y=global_y,
                    house_id=house.id,
                    age_group=age_group,
                    mask_status=mask_status,
                    infection_status=infection_status
                )
                
                people_list.append(person)
                person_id_counter += 1
                break
                
    return people_list

def visualize_map(map_array, people_list=None):
    """
    Visualize the generated map with people overlaid
    
    Parameters:
    - map_array: The 2D numpy array representing the map.
    - people_list: Optional list of Person objects to visualize
    """
    # Create a copy of the map to visualize (so we don't modify the original)
    # viz_map = map_array.copy()
    
    # 보기 편하도록 숫자 변경
    # house -> 2
    # building -> 3
    # road -> 1
    # empty space -> 0
    # viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
    # viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
    
    # Add people to the visualization if provided
    # if people_list:
    #     for person in people_list:
            # Different colors for different infection statuses
            # 5: healthy, 6: infected, 7: recovered, 8: dead
            # viz_map[person.x, person.y] = 5 + person.infection_status
    
    # plt.figure(figsize=(10, 10))
    
    # Create a custom colormap for better visualization
    # import matplotlib.colors as mcolors
    # colors = ['white', 'gray', 'green', 'blue', 'black', 'yellow', 'red', 'purple', 'black']
    # cmap = mcolors.ListedColormap(colors)
    
    # Set bounds for colorbar
    # bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # plt.imshow(viz_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Create custom colorbar
    # cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5])
    # cbar.ax.set_yticklabels(['Empty', 'Road', 'House', 'Building', 'Healthy', 'Infected', 'Recovered', 'Dead'])
    
    # plt.title('COVID-19 Simulation Map')
    # plt.show()

def save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir='./'):
    """
    Save the simulation state to NPY files.
    
    Parameters:
    - map_array: The map grid
    - people_list: List of Person objects
    - houses_list: List of House objects
    - buildings_list: List of Building objects
    - npy_save_dir: Directory to save the NPY files
    """
    # Ensure the output directory exists
    if not os.path.exists(npy_save_dir):
        os.makedirs(npy_save_dir)
    
    # Get current timestamp for filename prefix
    timestamp = str(int(time.time() * 1000))
    
    # Save the map array
    np.save(os.path.join(npy_save_dir, f'{timestamp}_map_array.npy'), map_array)
    
    # Convert people_list to a structured array for saving
    people_data = []
    for person in people_list:
        people_data.append({
            'id': person.id,
            'x': person.x, 
            'y': person.y,
            'house_id': person.house_id,
            'age_group': person.age_group,
            'mask_status': person.mask_status,
            'infection_status': person.infection_status,
            'dest_type': 0 if person.dest_type is None else (1 if person.dest_type == 'house' else 2),
            'dest_id': -1 if person.dest_id is None else person.dest_id
        })
    
    # Convert to numpy structured array and save
    people_dtype = [
        ('id', int), ('x', int), ('y', int), ('house_id', int), 
        ('age_group', int), ('mask_status', int), ('infection_status', int),
        ('dest_type', int), ('dest_id', int)
    ]
    # Convert list of dictionaries to list of tuples in the right order
    people_tuples = [(p['id'], p['x'], p['y'], p['house_id'], p['age_group'], 
                      p['mask_status'], p['infection_status'], p['dest_type'], p['dest_id']) 
                     for p in people_data]
    people_array = np.array(people_tuples, dtype=people_dtype)
    np.save(os.path.join(npy_save_dir, f'{timestamp}_people.npy'), people_array)
    
    # Save houses data
    houses_data = []
    for house in houses_list:
        houses_data.append({
            'id': house.id,
            'x': house.x,
            'y': house.y,
            'size': house.size
        })
    
    houses_dtype = [('id', int), ('x', int), ('y', int), ('size', int)]
    # Convert list of dictionaries to list of tuples in the right order
    houses_tuples = [(h['id'], h['x'], h['y'], h['size']) for h in houses_data]
    houses_array = np.array(houses_tuples, dtype=houses_dtype)
    np.save(os.path.join(npy_save_dir, f'{timestamp}_houses.npy'), houses_array)
    
    # Save buildings data
    buildings_data = []
    for building in buildings_list:
        buildings_data.append({
            'id': building.id,
            'x': building.x,
            'y': building.y,
            'size': building.size
        })
    
    buildings_dtype = [('id', int), ('x', int), ('y', int), ('size', int)]
    # Convert list of dictionaries to list of tuples in the right order
    buildings_tuples = [(b['id'], b['x'], b['y'], b['size']) for b in buildings_data]
    buildings_array = np.array(buildings_tuples, dtype=buildings_dtype)
    np.save(os.path.join(npy_save_dir, f'{timestamp}_buildings.npy'), buildings_array)
    
    print(f"Simulation state saved to {npy_save_dir} with timestamp {timestamp}")

def run_simulation(map_array, people_list, houses_list, buildings_list, num_days=100, infection_rate=[0.3, 0.01, 0.001], recovery_rate=[0.1, 0.05, 0.02], death_rate=[0.01, 0.005, 0.02], infected_people_conscience=0.7, npy_save_dir='./', timing_output_file=None):
    """
    Run a COVID-19 simulation
    
    Parameters:
    - map_array: The map grid
    - people_list: List of Person objects
    - houses_list: List of House objects
    - buildings_list: List of Building objects
    - num_steps: Number of time steps to simulate
    - infection_rate: List of infection rates for different mask statuses (no_mask, cloth_mask, mask+face_shield)
    - recovery_rate: List of recovery rates for different age groups (child, adult, senior)
    - death_rate: List of death rates for different age groups (child, adult, senior)
    - timing_output_file: File to save timing results
    """
    # Initialize timing manager
    timing = TimingManager(output_file=timing_output_file, rank=0)
    timing.start_timing("total_simulation")
    
    # Ensure the output directory exists
    if not os.path.exists(npy_save_dir):
        os.makedirs(npy_save_dir)

    # This is a placeholder for the actual simulation logic
    # You would implement movement, infection spread, etc. here
    print(f"Starting simulation with {len(people_list)} people...")
    
    # Example: Count initial infections
    infected_count = sum(1 for person in people_list if person.infection_status == 1)
    print(f"Initial infected count: {infected_count}")
    
    # Visualize initial state
    visualize_map(map_array, people_list)
    
    # Placeholder for simulation steps
    for day in range(num_days):
        timing.start_timing(f"day_{day}")
        print(f"[TIMING] Starting day {day}")
        # Simulate each day
        # First step, update people going to their destinations
        for person in people_list:
            person.dest_type = None  # Reset destination
            person.dest_id = None
            
            # Skip dead people
            if person.infection_status == 3:  # Dead status
                continue
                
            # If infected, they might stay home based on conscience
            if person.infection_status == 1 and random.random() < infected_people_conscience:
                # Infected people might not go to work
                continue
                
            # All other people (healthy, recovered, or infected without conscience)
            # will go to work
            person.dest_type = 'building'
            person.dest_id = random.choice(buildings_list).id

        # day is started, move move move move
        while True:
            # Save current state of people for visualization
            save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir)
            # Simulate movement of people
            simulate_movement(people_list, map_array, houses_list, buildings_list)
            # Simulate infection spread on roads during movement
            simulate_infection(people_list, infection_rate, recovery_rate, death_rate)
            # Visualize after each movement
            # visualize_map(map_array, people_list)
            
            # Check if all people have reached their destinations
            if all(person.dest_type is None for person in people_list):
                break
                
        timing.start_timing(f"day_{day}_data_saving")
        save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir)
        elapsed_save = timing.end_timing(f"day_{day}_data_saving")
        print(f"[TIMING] Data saving: {elapsed_save:.4f} seconds")

        print(f"Day {day + 1} everyone has reached their work or school or sum idk.")

        # Simulate infection spread inside buildings
        timing.start_timing(f"day_{day}_infection_inside_buildings")
        simulate_infection_inside(people_list, houses_list, buildings_list, 
                               infection_rate, recovery_rate, death_rate, 
                               simulation_iterations=3)  # More iterations for workplaces
        elapsed_infection = timing.end_timing(f"day_{day}_infection_inside_buildings")
        print(f"[TIMING] Infection simulation in buildings: {elapsed_infection:.4f} seconds")


        # And go home yeyeyeyeyeyeyeyeyeyeyeyeyey   
        for person in people_list:
            person.dest_type = 'house'
            person.dest_id = person.house_id  # Go back to their house

        while True:
            # Save current state of people for visualization
            save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir)
            # Simulate movement of people
            simulate_movement(people_list, map_array, houses_list, buildings_list)
            # Simulate infection spread on roads during movement
            simulate_infection(people_list, infection_rate, recovery_rate, death_rate)
            # Visualize after each movement
            # visualize_map(map_array, people_list)

            # Check if all people have reached their destinations
            if all(person.dest_type is None for person in people_list):
                break
        # visualize_map(map_array, people_list)

        save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir)
        
        # Simulate infection spread inside houses
        timing.start_timing(f"day_{day}_infection_inside_houses")
        simulate_infection_inside(people_list, houses_list, buildings_list, 
                               infection_rate, recovery_rate, death_rate, 
                               simulation_iterations=5)  # More iterations for home as people spend more time there
        elapsed_infection = timing.end_timing(f"day_{day}_infection_inside_houses")
        print(f"[TIMING] Infection simulation in houses: {elapsed_infection:.4f} seconds")
        # Simulate death based on infection status
        for person in people_list:
            if person.infection_status == 1:
                # Simulate death based on death rate
                if random.random() < death_rate[person.age_group]:
                    person.infection_status = 3

        save_npy(map_array, people_list, houses_list, buildings_list, npy_save_dir)

        print(f"Day {day + 1} everyone has reached their home.")
        print(f"Day {day + 1} simulation step completed.")

        print(f"Day {day + 1} statistics:")
        # Report statistics
        infected_count = sum(1 for person in people_list if person.infection_status == 1)
        recovered_count = sum(1 for person in people_list if person.infection_status == 2)
        dead_count = sum(1 for person in people_list if person.infection_status == 3)
        print(f"Infected: {infected_count}")
        print(f"Recovered: {recovered_count}")
        print(f"Dead: {dead_count}")
        print(f"Day {day + 1} everyone has reached their home.")
        
        # End timing for this day
        elapsed_day = timing.end_timing(f"day_{day}")
        print(f"[TIMING] Day {day} completed in: {elapsed_day:.4f} seconds")

    # Visualize final state
    visualize_map(map_array, people_list)
    
    # Report final statistics
    infected_count = sum(1 for person in people_list if person.infection_status == 1)
    recovered_count = sum(1 for person in people_list if person.infection_status == 2)
    dead_count = sum(1 for person in people_list if person.infection_status == 3)
    
    print(f"Final statistics after {num_days} days:")
    print(f"Infected: {infected_count}")
    print(f"Recovered: {recovered_count}")
    print(f"Dead: {dead_count}")
    
    # End total timing and print summary
    elapsed_total = timing.end_timing("total_simulation")
    print(f"[TIMING] Total sequential simulation time: {elapsed_total:.4f} seconds")
    timing.print_all_timings()
    
    # Save timing results to file
    if timing_output_file:
        timing.save_timings()

def find_path(map_array, start_x, start_y, target_x, target_y):
    """
    Find the shortest path on roads from start to target using BFS.
    
    Parameters:
    - map_array: The map grid
    - start_x, start_y: Starting position
    - target_x, target_y: Target position
    
    Returns:
    - A list of (x, y) coordinates representing the path from start to target,
      or None if no path is found
    """
    # Check if start or target is not a road
    if start_x < 0 or start_y < 0 or start_x >= map_array.shape[0] or start_y >= map_array.shape[1]:
        return None
        
    # Directions: up, right, down, left
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    
    # Queue for BFS
    queue = [(start_x, start_y)]
    # Visited set to avoid cycles
    visited = set([(start_x, start_y)])
    # Parent dictionary to reconstruct the path
    parent = {}
    
    while queue:
        x, y = queue.pop(0)
        
        # Check if we've reached the target
        if x == target_x and y == target_y:
            # Reconstruct path
            path = []
            current = (x, y)
            while current != (start_x, start_y):
                path.append(current)
                current = parent[current]
            path.append((start_x, start_y))
            # Return path in correct order (start to target)
            return path[::-1]
        
        # Try all four directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if within bounds
            if 0 <= nx < map_array.shape[0] and 0 <= ny < map_array.shape[1]:
                # Check if it's a road and not visited
                if map_array[nx, ny] == 1 and (nx, ny) not in visited:
                    queue.append((nx, ny))
                    visited.add((nx, ny))
                    parent[(nx, ny)] = (x, y)
    
    # No path found
    return None

def find_nearest_road(map_array, target_x, target_y):
    """
    Find the nearest road to a target point.
    
    Parameters:
    - map_array: The map grid
    - target_x, target_y: Target position
    
    Returns:
    - (x, y) of the nearest road point, or None if not found
    """
    # Check if target is already a road
    if (0 <= target_x < map_array.shape[0] and 0 <= target_y < map_array.shape[1] and 
        map_array[target_x, target_y] == 1):
        return (target_x, target_y)
    
    # BFS to find nearest road
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    queue = [(target_x, target_y, 0)]  # (x, y, distance)
    visited = set([(target_x, target_y)])
    
    while queue:
        x, y, dist = queue.pop(0)
        
        # Check all four directions
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if within bounds
            if 0 <= nx < map_array.shape[0] and 0 <= ny < map_array.shape[1]:
                # Check if it's a road
                if map_array[nx, ny] == 1:
                    return (nx, ny)
                
                # Add to queue if not visited
                if (nx, ny) not in visited:
                    queue.append((nx, ny, dist + 1))
                    visited.add((nx, ny))
    
    # No road found
    return None

def norm_cdf(x):
    """
    Approximation of the cumulative distribution function for the standard normal distribution.
    """
    # Constants for approximation
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    p = 0.2316419
    c = 0.39894228
    
    # Ensure x is positive
    if x >= 0:
        t = 1.0 / (1.0 + p * x)
        return 1.0 - c * np.exp(-x * x / 2.0) * (((((b5 * t + b4) * t + b3) * t + b2) * t + b1) * t)
    else:
        # Use symmetry property
        return 1.0 - norm_cdf(-x)

def simulate_movement(people_list, map_array, houses_list, buildings_list):
    """
    Simulate movement of people on the map using pathfinding.
    
    Parameters:
    - people_list: List of Person objects
    - map_array: The map grid
    - houses_list: List of House objects
    - buildings_list: List of Building objects
    """
    for person in people_list:
        # check if the person has a destination
        if person.dest_type is None or person.dest_id is None:
            # Person has no destination, they might be inside their house or building
            continue
            
        # check if the person is at their destination
        # find the house x y range with the destination ID
        if person.dest_type == 'house':
            destination = next((house for house in houses_list if house.id == person.dest_id), None)
            # check if the person is at their destination
            if destination and destination.contains_point(person.x, person.y):
                # Person has reached their house, reset destination
                person.dest_type = None
                person.dest_id = None
                continue
        elif person.dest_type == 'building':
            destination = next((building for building in buildings_list if building.id == person.dest_id), None)
            # check if the person is at their destination
            if destination and destination.contains_point(person.x, person.y):
                # Person has reached their building, reset destination
                person.dest_type = None
                person.dest_id = None
                continue
        
        # check if the person is inside a house or building (that is not their destination)
        if map_array[person.x, person.y] in range(200, 400):
            # Person is inside a house that is not their destination
            # Find an exit to a road
            
            # 건물의 종류 확인 (집인지 빌딩인지)
            building_id = map_array[person.x, person.y] - 200 if map_array[person.x, person.y] < 300 else map_array[person.x, person.y] - 300
            
            # 건물 정보 가져오기
            if map_array[person.x, person.y] < 300:  # 집인 경우
                building = next((h for h in houses_list if h.id == building_id), None)
            else:  # 빌딩인 경우
                building = next((b for b in buildings_list if b.id == building_id), None)
            
            if building:
                # 건물의 경계 확인
                min_x, max_x = building.x, building.x + building.size - 1
                min_y, max_y = building.y, building.y + building.size - 1
                
                # 건물 경계에 인접한 도로 찾기
                possible_exits = []
                
                # 건물의 모든 가장자리를 확인
                # 위쪽 가장자리
                for x in range(min_x, max_x + 1):
                    # 위쪽에 도로가 있는지 확인
                    if min_y > 0 and map_array[x, min_y - 1] == 1:
                        possible_exits.append((x, min_y - 1))
                
                # 아래쪽 가장자리
                for x in range(min_x, max_x + 1):
                    # 아래쪽에 도로가 있는지 확인
                    if max_y < map_array.shape[0] - 1 and map_array[x, max_y + 1] == 1:
                        possible_exits.append((x, max_y + 1))
                
                # 왼쪽 가장자리
                for y in range(min_y, max_y + 1):
                    # 왼쪽에 도로가 있는지 확인
                    if min_x > 0 and map_array[min_x - 1, y] == 1:
                        possible_exits.append((min_x - 1, y))
                
                # 오른쪽 가장자리
                for y in range(min_y, max_y + 1):
                    # 오른쪽에 도로가 있는지 확인
                    if max_x < map_array.shape[1] - 1 and map_array[max_x + 1, y] == 1:
                        possible_exits.append((max_x + 1, y))

                # 가능한 출구가 있으면 하나 선택해서 이동
                if possible_exits:
                    exit_x, exit_y = random.choice(possible_exits)
                    person.x, person.y = exit_x, exit_y
            continue

        # Find target position
        target = None
        if person.dest_type == 'house':
            target = next((house for house in houses_list if house.id == person.dest_id), None)
        elif person.dest_type == 'building':
            target = next((building for building in buildings_list if building.id == person.dest_id), None)
        else:
            continue  # No valid destination
            
        if not target:
            continue
            
        # Get center of target for pathfinding reference
        target_center_x, target_center_y = target.x + target.size // 2, target.y + target.size // 2
        
        # Find a road point near the target if target is not on a road
        nearest_road = find_nearest_road(map_array, target_center_x, target_center_y)
        if not nearest_road:
            continue
        
        target_road_x, target_road_y = nearest_road
        
        # Check if we're on a road
        if map_array[person.x, person.y] != 1:
            # Not on a road - shouldn't happen, but just in case
            continue

        # Check if we're adjacent to the target
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        is_adjacent_to_dest = False
        
        for dx, dy in directions:
            nx, ny = person.x + dx, person.y + dy
            if (0 <= nx < map_array.shape[0] and 0 <= ny < map_array.shape[1] and 
                target.contains_point(nx, ny)):
                is_adjacent_to_dest = True
                break
                
        # If we're adjacent to the target, enter it
        if is_adjacent_to_dest:
            valid_positions = []
            for x in range(target.x, target.x + target.size):
                for y in range(target.y, target.y + target.size):
                    valid_positions.append((x, y))
            
            if valid_positions:
                # Choose a random valid position inside the building
                best_move = random.choice(valid_positions)
                person.x, person.y = best_move
                continue
        
        # Find the shortest path on roads to the target
        path = find_path(map_array, person.x, person.y, target_road_x, target_road_y)
        
        # If path found, move along the path (one step at a time)
        if path and len(path) > 1:  # Need at least current position and next position
            # The next position in the path is at index 1 (index 0 is current position)
            next_x, next_y = path[1]
            person.x, person.y = next_x, next_y

def simulate_infection(people_list, infection_rate, recovery_rate, death_rate):
    """
    Simulate infection spread among people on the roads.
    
    Parameters:
    - people_list: List of Person objects
    - infection_rate: Infection rate for different mask statuses
    - recovery_rate: Recovery rate for different age groups
    - death_rate: Death rate for different age groups
    """
    # Find all infected people (infection_status == 1)
    infected_people = [person for person in people_list if person.infection_status == 1]
    
    # For each infected person, check for potential transmission to others
    for infected in infected_people:
        # Skip dead people
        if infected.infection_status == 3:
            continue
            
        # Get infected person's mask status to determine base transmission rate
        base_rate = infection_rate[infected.mask_status]
        
        # Check all other people for potential infection
        for person in people_list:
            # Skip if the person is already infected, recovered, dead, or is the infector
            if (person.id == infected.id or 
                person.infection_status != 0):  # Not normal health status
                continue
                
            # Calculate distance between infected and potential victim
            dx = infected.x - person.x
            dy = infected.y - person.y
            distance = (dx ** 2 + dy ** 2) ** 0.5  # Euclidean distance
            
            # If they're too far apart, no infection chance
            if distance > 5:  # Arbitrary cutoff distance for efficiency
                continue
                
            # Calculate infection probability based on distance
            if distance == 0:  # Same position
                distance_factor = 1.5
            elif distance == 1:  # Adjacent (orthogonal)
                distance_factor = 1.0
            else:
                # For distance > 1, use normal distribution to decay probability
                # n-1 on standard normal distribution table, multiply by 2 to get a steeper falloff
                std_normal_value = norm_cdf(distance - 1)
                distance_factor = 1 - (1 - std_normal_value) * 2
                
                # Ignore very small probabilities for efficiency
                if distance_factor < 0.01:
                    continue
            
            # Final infection probability
            infection_probability = base_rate * distance_factor
            
            # Determine if infection occurs
            if random.random() < infection_probability:
                person.infection_status = 1  # Set to infected

def simulate_infection_inside(people_list, houses_list, buildings_list, infection_rate, recovery_rate, death_rate, simulation_iterations=1):
    """
    Simulate infection spread inside houses and buildings.
    
    Parameters:
    - people_list: List of Person objects
    - houses_list: List of House objects
    - buildings_list: List of Building objects
    - infection_rate: Infection rate for different mask statuses
    - recovery_rate: Recovery rate for different age groups
    - death_rate: Death rate for different age groups
    - simulation_iterations: Number of infection simulation iterations within buildings
    """
    # Group people by their current location (house or building)
    people_by_house = {}
    people_by_building = {}
    
    for person in people_list:
        # Skip dead people
        if person.infection_status == 3:
            continue
            
        # Check if person is in a house
        for house in houses_list:
            if house.contains_point(person.x, person.y):
                if house.id not in people_by_house:
                    people_by_house[house.id] = []
                people_by_house[house.id].append(person)
                break
        
        # Check if person is in a building
        for building in buildings_list:
            if building.contains_point(person.x, person.y):
                if building.id not in people_by_building:
                    people_by_building[building.id] = []
                people_by_building[building.id].append(person)
                break
    
    # For each building, run multiple infection simulation iterations
    for _ in range(simulation_iterations):
        # Process infections in houses
        for house_id, occupants in people_by_house.items():
            # Find infected occupants
            infected_occupants = [p for p in occupants if p.infection_status == 1]
            
            # If there are infected people in the house
            for infected in infected_occupants:
                # Get base infection rate based on infected person's mask status
                base_rate = infection_rate[infected.mask_status]
                
                # Increased transmission rate for indoor environments
                indoor_factor = 2.0  # Transmission is higher indoors
                
                # Check for transmission to each other occupant
                for person in occupants:
                    # Skip if the person is already infected, recovered, dead, or is the infector
                    if (person.id == infected.id or 
                        person.infection_status != 0):
                        continue
                    
                    # Higher chance of infection indoors
                    infection_probability = base_rate * indoor_factor
                    
                    # Determine if infection occurs
                    if random.random() < infection_probability:
                        person.infection_status = 1  # Set to infected
        
        # Process infections in buildings (same logic as houses, but separate loop)
        for building_id, occupants in people_by_building.items():
            # Find infected occupants
            infected_occupants = [p for p in occupants if p.infection_status == 1]
            
            # If there are infected people in the building
            for infected in infected_occupants:
                # Get base infection rate based on infected person's mask status
                base_rate = infection_rate[infected.mask_status]
                
                # Increased transmission rate for indoor environments
                indoor_factor = 1.5  # Transmission is higher indoors, but less than houses due to potentially larger space
                
                # Check for transmission to each other occupant
                for person in occupants:
                    # Skip if the person is already infected, recovered, dead, or is the infector
                    if (person.id == infected.id or 
                        person.infection_status != 0):
                        continue
                    
                    # Higher chance of infection indoors
                    infection_probability = base_rate * indoor_factor
                    
                    # Determine if infection occurs
                    if random.random() < infection_probability:
                        person.infection_status = 1  # Set to infected
    
    # Process recovery and death after infection spread is simulated
    for person in people_list:
        # Skip non-infected people
        if person.infection_status != 1:
            continue
            
        # Check for recovery based on age group
        if random.random() < recovery_rate[person.age_group]:
            person.infection_status = 2  # Recovered

def split_people_by_regions(people_list, map_size, num_processes):
    """
    Split people into regions for parallel processing.
    
    Parameters:
    - people_list: List of Person objects
    - map_size: Size of the map (assuming it's square)
    - num_processes: Number of MPI processes
    
    Returns:
    - local_people: List of people in this process's region
    """
    # MPI rank and size
    comm = MPI.COMM_WORLD 
    rank = comm.Get_rank()
    
    # Calculate region boundaries for each process (divide map into rows)
    region_height = map_size // num_processes
    start_row = rank * region_height
    end_row = (rank + 1) * region_height if rank < num_processes - 1 else map_size
    
    # Filter people in this process's region
    local_people = [person for person in people_list if start_row <= person.x < end_row]
    
    print(f"[DEBUG][Rank {rank}] Region: rows {start_row}-{end_row}, Got {len(local_people)} people out of {len(people_list)}")
    
    return local_people, start_row, end_row

def exchange_boundary_data(local_people, start_row, end_row, map_size):
    """
    Exchange people who moved across process boundaries.
    
    Parameters:
    - local_people: List of Person objects in this process's region
    - start_row: Starting row of this process's region
    - end_row: Ending row of this process's region
    - map_size: Size of the map
    
    Returns:
    - Updated local_people list after boundary exchange
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # People moving out of this process's region
    outgoing_people = [p for p in local_people if p.x < start_row or p.x >= end_row]
    
    # People staying in this process's region
    staying_people = [p for p in local_people if start_row <= p.x < end_row]
    
    print(f"[DEBUG][Rank {rank}] Exchange: {len(outgoing_people)} people leaving, {len(staying_people)} staying")
    
    # 더 안전한 점대점 통신 방식으로 변경
    updated_local_people = staying_people[:]  # 복사본 생성
    
    # 각 프로세스별로 필요한 사람들을 직접 교환
    for target_rank in range(size):
        # 현재 프로세스에서 target_rank로 보낼 사람들 필터링
        region_start = target_rank * (map_size // size)
        region_end = (target_rank + 1) * (map_size // size) if target_rank < size - 1 else map_size
        send_people = [p for p in outgoing_people if region_start <= p.x < region_end]
        
        # 디버그 정보
        if send_people:
            print(f"[DEBUG][Rank {rank}] Sending {len(send_people)} people to rank {target_rank}")
            for person in send_people:
                print(f"[DEBUG][Rank {rank}] Person {person.id} at ({person.x},{person.y}) moving to rank {target_rank}")
        
        # 통신 객체 생성: 리스트 타입
        send_obj = send_people
        
        try:
            # 동기화 지점을 하나씩 처리: 한 번에 한 쌍의 프로세스만 통신
            # (내 프로세스, 타겟 프로세스) 쌍
            if rank == target_rank:
                # 자기 자신과는 교환할 필요 없음
                recv_obj = []
            else:
                print(f"[DEBUG][Rank {rank}] Exchanging data with rank {target_rank}")
                # 교착상태를 방지하기 위한 알고리즘 개선
                # 낮은 랭크에서 높은 랭크로는 먼저 보내고 받기, 높은 랭크에서 낮은 랭크로는 먼저 받고 보내기
                if rank < target_rank:
                    # 낮은 랭크: 먼저 보내고 나중에 받음
                    comm.send(send_obj, dest=target_rank, tag=10*rank+target_rank)
                    recv_obj = comm.recv(source=target_rank, tag=10*target_rank+rank)
                else:
                    # 높은 랭크: 먼저 받고 나중에 보냄
                    recv_obj = comm.recv(source=target_rank, tag=10*target_rank+rank)
                    comm.send(send_obj, dest=target_rank, tag=10*rank+target_rank)
                
                print(f"[DEBUG][Rank {rank}] Received {len(recv_obj)} people from rank {target_rank}")
                
            # 받은 사람들을 로컬 리스트에 추가
            updated_local_people.extend(recv_obj)
            
        except Exception as e:
            print(f"[ERROR][Rank {rank}] Exception during exchange with rank {target_rank}: {str(e)}")
    
    # 동기화를 위한 배리어 추가
    try:
        comm.Barrier()
        print(f"[DEBUG][Rank {rank}] Exchange completed: now has {len(updated_local_people)} people")
        return updated_local_people
    except Exception as e:
        print(f"[ERROR][Rank {rank}] Exception during barrier after exchange: {str(e)}")
        return staying_people

def gather_all_people(local_people):
    """
    Gather all people from all processes.
    
    Parameters:
    - local_people: List of Person objects in this process's region
    
    Returns:
    - all_people: Combined list of all people from all processes (only on rank 0)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    print(f"[DEBUG][Rank {rank}] Starting gather with {len(local_people)} local people")
    
    # gather 대신 점대점 통신을 사용하여 rank 0에 모든 데이터 수집
    try:
        if rank == 0:
            # rank 0은 모든 다른 프로세스로부터 데이터를 수집
            all_people = [local_people[:]]  # 자신의 데이터로 초기화
            
            # 다른 모든 프로세스에서 데이터 수집
            for source_rank in range(1, size):
                print(f"[DEBUG][Rank 0] Waiting to receive data from rank {source_rank}")
                people_from_rank = comm.recv(source=source_rank, tag=source_rank)
                print(f"[DEBUG][Rank 0] Received {len(people_from_rank)} people from rank {source_rank}")
                all_people.append(people_from_rank)
            
            # 결과 병합
            flattened = [person for sublist in all_people for person in sublist]
            print(f"[DEBUG][Rank 0] Gather complete: collected {len(flattened)} people total")
            return flattened
        else:
            # 다른 프로세스는 rank 0으로 데이터를 전송
            print(f"[DEBUG][Rank {rank}] Sending {len(local_people)} people to rank 0")
            comm.send(local_people, dest=0, tag=rank)
            print(f"[DEBUG][Rank {rank}] Data sent to rank 0")
            return None
            
    except Exception as e:
        print(f"[ERROR][Rank {rank}] Exception during gather: {str(e)}")
        if rank == 0:
            print("[WARNING][Rank 0] Returning local people only due to gather failure")
            return local_people
        return None

def run_simulation_parallel(map_array, people_list, houses_list, buildings_list, num_days=100, 
                           infection_rate=[0.3, 0.01, 0.001], recovery_rate=[0.1, 0.05, 0.02], 
                           death_rate=[0.01, 0.005, 0.02], infected_people_conscience=0.7, 
                           npy_save_dir='./', timing_output_file=None):
    """
    Run a COVID-19 simulation in parallel using MPI
    """
    # MPI initialization
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Initialize timing manager
    timing = TimingManager(output_file=timing_output_file, rank=rank)
    timing.start_timing("total_simulation")
    
    print(f"[DEBUG][Rank {rank}] Starting parallel simulation, rank {rank} of {size}")
    
    # Map dimensions
    map_size = map_array.shape[0]
    
    # Ensure the output directory exists (only on rank 0)
    if rank == 0:
        if not os.path.exists(npy_save_dir):
            os.makedirs(npy_save_dir)
        print(f"Starting parallel simulation with {len(people_list)} people on {size} processes...")
        
        # Count initial infections
        infected_count = sum(1 for person in people_list if person.infection_status == 1)
        print(f"Initial infected count: {infected_count}")
    
    # Split people by regions for initial distribution
    print(f"[DEBUG][Rank {rank}] Splitting people by regions")
    try:
        timing.start_timing("split_people_by_regions")
        local_people, start_row, end_row = split_people_by_regions(people_list, map_size, size)
        elapsed = timing.end_timing("split_people_by_regions")
        print(f"[TIMING][Rank {rank}] Split people by regions: {elapsed:.4f} seconds")
    except Exception as e:
        print(f"[ERROR][Rank {rank}] Error during splitting: {str(e)}")
        return
    
    # Synchronize to ensure all processes have their people assigned
    print(f"[DEBUG][Rank {rank}] Waiting at barrier before simulation loop")
    try:
        comm.Barrier()
        print(f"[DEBUG][Rank {rank}] Barrier passed, starting simulation loop")
    except Exception as e:
        print(f"[ERROR][Rank {rank}] Error at barrier: {str(e)}")
        return
    
    for day in range(num_days):
        timing.start_timing(f"day_{day}")
        print(f"[TIMING][Rank {rank}] Starting day {day}")
        
        # First step, update people going to their destinations
        timing.start_timing(f"day_{day}_destination_assignment")
        for person in local_people:
            person.dest_type = None  # Reset destination
            person.dest_id = None
            
            # Skip dead people
            if person.infection_status == 3:  # Dead status
                continue
                
            # If infected, they might stay home based on conscience
            if person.infection_status == 1 and random.random() < infected_people_conscience:
                # Infected people might not go to work
                continue
                
            # All other people will go to work
            person.dest_type = 'building'
            person.dest_id = random.choice(buildings_list).id

        elapsed = timing.end_timing(f"day_{day}_destination_assignment")
        print(f"[TIMING][Rank {rank}][Day {day}] Destination assignment: {elapsed:.4f} seconds")

        # Day is started, move people
        print(f"[DEBUG][Rank {rank}][Day {day}] Starting movement cycle")
        moving = True
        while moving:
            # Simulate movement of people
            print(f"[DEBUG][Rank {rank}][Day {day}] Simulating movement for {len(local_people)} people")
            try:
                timing.start_timing(f"day_{day}_movement")
                simulate_movement(local_people, map_array, houses_list, buildings_list)
                elapsed = timing.end_timing(f"day_{day}_movement")
                print(f"[TIMING][Rank {rank}][Day {day}] Movement simulation: {elapsed:.4f} seconds")
                print(f"[DEBUG][Rank {rank}][Day {day}] Movement simulation complete")
            except Exception as e:
                print(f"[ERROR][Rank {rank}][Day {day}] Error during movement simulation: {str(e)}")
            
            # Exchange people who crossed boundaries
            print(f"[DEBUG][Rank {rank}][Day {day}] Starting boundary exchange")
            try:
                timing.start_timing(f"day_{day}_boundary_exchange")
                local_people = exchange_boundary_data(local_people, start_row, end_row, map_size)
                elapsed = timing.end_timing(f"day_{day}_boundary_exchange")
                print(f"[TIMING][Rank {rank}][Day {day}] Boundary exchange: {elapsed:.4f} seconds")
                print(f"[DEBUG][Rank {rank}][Day {day}] Boundary exchange complete")
            except Exception as e:
                print(f"[ERROR][Rank {rank}][Day {day}] Error during boundary exchange: {str(e)}")
            
            # Simulate infection spread on roads during movement
            print(f"[DEBUG][Rank {rank}][Day {day}] Simulating infections")
            try:
                timing.start_timing(f"day_{day}_infection")
                simulate_infection(local_people, infection_rate, recovery_rate, death_rate)
                elapsed = timing.end_timing(f"day_{day}_infection")
                print(f"[TIMING][Rank {rank}][Day {day}] Infection simulation: {elapsed:.4f} seconds")
                print(f"[DEBUG][Rank {rank}][Day {day}] Infection simulation complete")
            except Exception as e:
                print(f"[ERROR][Rank {rank}][Day {day}] Error during infection simulation: {str(e)}")
            
            # Check if all people have reached their destinations
            print(f"[DEBUG][Rank {rank}][Day {day}] Checking if all people have arrived")
            local_all_arrived = all(person.dest_type is None for person in local_people)
            print(f"[DEBUG][Rank {rank}][Day {day}] Local arrival status: {local_all_arrived}")
            
            # Check if all processes have all their people arrived
            # allreduce 대신 점대점 통신을 사용하여 모든 도착 상태 확인
            print(f"[DEBUG][Rank {rank}][Day {day}] Checking global arrival status using point-to-point")
            timing.start_timing(f"day_{day}_communication")
            try:
                if rank == 0:
                    # 랭크 0은 모든 프로세스로부터 도착 상태를 수집
                    all_statuses = [local_all_arrived]  # 자신의 상태로 초기화
                    
                    # 다른 모든 프로세스에서 상태 수집
                    for source_rank in range(1, size):
                        print(f"[DEBUG][Rank 0][Day {day}] Waiting for arrival status from rank {source_rank}")
                        status_from_rank = comm.recv(source=source_rank, tag=500+source_rank)
                        print(f"[DEBUG][Rank 0][Day {day}] Received arrival status {status_from_rank} from rank {source_rank}")
                        all_statuses.append(status_from_rank)
                    
                    # 모든 프로세스가 완료되었는지 확인
                    all_arrived = all(all_statuses)
                    print(f"[DEBUG][Rank 0][Day {day}] Global arrival status: {all_arrived}")
                    
                    # 결과를 모든 프로세스에게 브로드캐스트 (점대점으로)
                    for dest_rank in range(1, size):
                        comm.send(all_arrived, dest=dest_rank, tag=600+dest_rank)
                else:
                    # 다른 프로세스는 자신의 상태를 랭크 0에게 전송
                    print(f"[DEBUG][Rank {rank}][Day {day}] Sending arrival status {local_all_arrived} to rank 0")
                    comm.send(local_all_arrived, dest=0, tag=500+rank)
                    
                    # 랭크 0으로부터 전체 상태 수신
                    all_arrived = comm.recv(source=0, tag=600+rank)
                    print(f"[DEBUG][Rank {rank}][Day {day}] Global arrival status received from rank 0: {all_arrived}")
                
                elapsed = timing.end_timing(f"day_{day}_communication")
                print(f"[TIMING][Rank {rank}][Day {day}] Communication: {elapsed:.4f} seconds")
            except Exception as e:
                print(f"[ERROR][Rank {rank}][Day {day}] Error during arrival status check: {str(e)}")
                timing.end_timing(f"day_{day}_communication")  # Still end timing even if there was an error
                all_arrived = False  # 에러 발생시 계속 이동
            
            # Gather all people to rank 0 for saving state (could be optimized to reduce communication)
            if rank == 0:
                print(f"[DEBUG][Rank {rank}][Day {day}] Gathering people data for saving")
                try:
                    timing.start_timing(f"day_{day}_data_gathering")
                    all_people = gather_all_people(local_people)
                    elapsed_gather = timing.end_timing(f"day_{day}_data_gathering")
                    print(f"[TIMING][Rank {rank}][Day {day}] Data gathering: {elapsed_gather:.4f} seconds")
                    
                    timing.start_timing(f"day_{day}_data_saving")
                    save_npy(map_array, all_people, houses_list, buildings_list, npy_save_dir)
                    elapsed_save = timing.end_timing(f"day_{day}_data_saving")
                    print(f"[TIMING][Rank {rank}][Day {day}] Data saving: {elapsed_save:.4f} seconds")
                    
                    print(f"[DEBUG][Rank {rank}][Day {day}] Save complete")
                except Exception as e:
                    print(f"[ERROR][Rank {rank}][Day {day}] Error during gather or save: {str(e)}")
            
            # Exit loop if all people have reached destinations
            if all_arrived:
                print(f"[DEBUG][Rank {rank}][Day {day}] All people have reached destinations")
                moving = False
        
        if rank == 0:
            print(f"Day {day + 1} everyone has reached their work or school.")

        # Simulate infection spread inside buildings
        simulate_infection_inside(local_people, houses_list, buildings_list, 
                               infection_rate, recovery_rate, death_rate, 
                               simulation_iterations=3)

        # All people go home
        for person in local_people:
            person.dest_type = 'house'
            person.dest_id = person.house_id  # Go back to their house

        # People moving home
        moving = True
        while moving:
            # Simulate movement of people
            simulate_movement(local_people, map_array, houses_list, buildings_list)
            
            # Exchange people who crossed boundaries
            local_people = exchange_boundary_data(local_people, start_row, end_row, map_size)
            
            # Simulate infection spread on roads during movement
            simulate_infection(local_people, infection_rate, recovery_rate, death_rate)
            
            # Check if all people have reached their destinations
            local_all_arrived = all(person.dest_type is None for person in local_people)
            print(f"[DEBUG][Rank {rank}][Day {day}] Local arrival status (home): {local_all_arrived}")
            
            # 점대점 통신으로 모든 프로세스의 도착 상태 확인
            try:
                if rank == 0:
                    # 랭크 0은 모든 프로세스로부터 도착 상태를 수집
                    all_statuses = [local_all_arrived]  # 자신의 상태로 초기화
                    
                    # 다른 모든 프로세스에서 상태 수집
                    for source_rank in range(1, size):
                        print(f"[DEBUG][Rank 0][Day {day}] Waiting for home arrival status from rank {source_rank}")
                        status_from_rank = comm.recv(source=source_rank, tag=700+source_rank)
                        print(f"[DEBUG][Rank 0][Day {day}] Received home arrival status {status_from_rank} from rank {source_rank}")
                        all_statuses.append(status_from_rank)
                    
                    # 모든 프로세스가 완료되었는지 확인
                    all_arrived = all(all_statuses)
                    print(f"[DEBUG][Rank 0][Day {day}] Global home arrival status: {all_arrived}")
                    
                    # 결과를 모든 프로세스에게 브로드캐스트 (점대점으로)
                    for dest_rank in range(1, size):
                        comm.send(all_arrived, dest=dest_rank, tag=800+dest_rank)
                else:
                    # 다른 프로세스는 자신의 상태를 랭크 0에게 전송
                    print(f"[DEBUG][Rank {rank}][Day {day}] Sending home arrival status {local_all_arrived} to rank 0")
                    comm.send(local_all_arrived, dest=0, tag=700+rank)
                    
                    # 랭크 0으로부터 전체 상태 수신
                    all_arrived = comm.recv(source=0, tag=800+rank)
                    print(f"[DEBUG][Rank {rank}][Day {day}] Global home arrival status received from rank 0: {all_arrived}")
            except Exception as e:
                print(f"[ERROR][Rank {rank}][Day {day}] Error during home arrival status check: {str(e)}")
                all_arrived = False
            
            # Gather all people to rank 0 for saving state
            if rank == 0:
                print(f"[DEBUG][Rank 0][Day {day}] Gathering people for home status save")
                all_people = gather_all_people(local_people)
                save_npy(map_array, all_people, houses_list, buildings_list, npy_save_dir)
            
            # Exit loop if all people have reached destinations
            if all_arrived:
                print(f"[DEBUG][Rank {rank}][Day {day}] All people have returned home")
                moving = False
        
        # Simulate infection spread inside houses
        simulate_infection_inside(local_people, houses_list, buildings_list, 
                               infection_rate, recovery_rate, death_rate, 
                               simulation_iterations=5)
                               
        # Simulate death based on infection status
        for person in local_people:
            if person.infection_status == 1:
                # Simulate death based on death rate
                if random.random() < death_rate[person.age_group]:
                    person.infection_status = 3
        
        # 통계 수집을 위한 로컬 카운트 계산
        local_infected = sum(1 for person in local_people if person.infection_status == 1)
        local_recovered = sum(1 for person in local_people if person.infection_status == 2)
        local_dead = sum(1 for person in local_people if person.infection_status == 3)
        
        print(f"[DEBUG][Rank {rank}][Day {day}] Local counts: infected={local_infected}, recovered={local_recovered}, dead={local_dead}")
        
        # 집합 통신(reduce) 대신 점대점 통신을 사용하여 통계 수집
        try:
            if rank == 0:
                # 랭크 0은 자신의 카운트로 초기화
                total_infected = local_infected
                total_recovered = local_recovered
                total_dead = local_dead
                
                # 다른 모든 프로세스로부터 카운트 수집
                for source_rank in range(1, size):
                    print(f"[DEBUG][Rank 0][Day {day}] Waiting for statistics from rank {source_rank}")
                    stats_from_rank = comm.recv(source=source_rank, tag=900+source_rank)
                    print(f"[DEBUG][Rank 0][Day {day}] Received statistics from rank {source_rank}")
                    
                    # 받은 통계 합산
                    total_infected += stats_from_rank[0]
                    total_recovered += stats_from_rank[1]
                    total_dead += stats_from_rank[2]
            else:
                # 다른 프로세스는 랭크 0에게 통계 전송
                stats_package = [local_infected, local_recovered, local_dead]
                print(f"[DEBUG][Rank {rank}][Day {day}] Sending statistics to rank 0")
                comm.send(stats_package, dest=0, tag=900+rank)
                
                # 다른 랭크에서는 통계를 사용하지 않으므로 None으로 설정
                total_infected = None
                total_recovered = None
                total_dead = None
        except Exception as e:
            print(f"[ERROR][Rank {rank}][Day {day}] Error during statistics collection: {str(e)}")
        
        # Save state and print statistics on rank 0
        if rank == 0:
            all_people = gather_all_people(local_people)
            save_npy(map_array, all_people, houses_list, buildings_list, npy_save_dir)
            
            print(f"Day {day + 1} everyone has reached their home.")
            print(f"Day {day + 1} simulation step completed.")
            print(f"Day {day + 1} statistics:")
            print(f"Infected: {total_infected}")
            print(f"Recovered: {total_recovered}")
            print(f"Dead: {total_dead}")
            
            # End timing for this day
            elapsed_day = timing.end_timing(f"day_{day}")
            print(f"[TIMING][Rank {rank}][Day {day}] Total day time: {elapsed_day:.4f} seconds")
    
    # Final statistics
    if rank == 0:
        all_people = gather_all_people(local_people)
        
        # Report final statistics
        infected_count = sum(1 for person in all_people if person.infection_status == 1)
        recovered_count = sum(1 for person in all_people if person.infection_status == 2)
        dead_count = sum(1 for person in all_people if person.infection_status == 3)
        
        print(f"Final statistics after {num_days} days:")
        print(f"Infected: {infected_count}")
        print(f"Recovered: {recovered_count}")
        print(f"Dead: {dead_count}")
        
        # End total timing and print summary
        elapsed_total = timing.end_timing("total_simulation")
        print(f"[TIMING][Rank {rank}] Total simulation time: {elapsed_total:.4f} seconds")
        timing.print_all_timings()
        
        # Save timing results to file
        if timing_output_file:
            timing.save_timings()

if __name__ == "__main__":
    # Initialize MPI
    try:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print(f"[INIT] Process {rank} of {size} started")
    except Exception as e:
        print(f"[CRITICAL ERROR] MPI initialization failed: {str(e)}")
        import sys
        sys.exit(1)
    
    # Example usage
    map_size = 100
    
    # Only rank 0 generates the initial map and people
    if rank == 0:
        map_array, houses_list, buildings_list = generate_map(
            size=map_size, 
            houses=20, 
            house_size=3, 
            buildings=5, 
            building_size=5,
            roads=2,
        )
        
        # Generate people and place them in houses
        people_list = generate_people(
            map_array, 
            houses_list, 
            person_min_per_house=5, 
            person_max_per_house=8, 
            infection_rate=0.05,
            mask_distribution=[0.3, 0.6, 0.1],  # 30% no mask, 60% cloth mask, 10% mask+face shield
        )
        
        print(f"Map size: {map_size}x{map_size}")
        print(f"Houses: {len(houses_list)}")
        print(f"Buildings: {len(buildings_list)}")
        print(f"People: {len(people_list)}")
        
        # Print sample of people for debugging
        print("\nSample of people:")
        for i in range(min(5, len(people_list))):
            print(people_list[i])
    else:
        # Other ranks initialize with empty values
        map_array = None
        houses_list = None
        buildings_list = None
        people_list = None
    
    # Broadcast map and objects to all processes
    print(f"[DEBUG][Rank {rank}] Starting broadcast of data")
    try:
        print(f"[DEBUG][Rank {rank}] Broadcasting map_array")
        map_array = comm.bcast(map_array, root=0)
        print(f"[DEBUG][Rank {rank}] Broadcasting houses_list")
        houses_list = comm.bcast(houses_list, root=0)
        print(f"[DEBUG][Rank {rank}] Broadcasting buildings_list")
        buildings_list = comm.bcast(buildings_list, root=0)
        print(f"[DEBUG][Rank {rank}] Broadcasting people_list")
        people_list = comm.bcast(people_list, root=0)
        print(f"[DEBUG][Rank {rank}] All broadcasts complete")
    except Exception as e:
        print(f"[CRITICAL ERROR][Rank {rank}] Error during broadcast: {str(e)}")
        import sys
        sys.exit(1)
    
    # Create output directory for simulation data
    timestamp = int(time.time())
    save_dir = f'./simulation_data_{timestamp}/'
    
    # Create timing output directory
    timing_dir = './timing_results'
    if rank == 0 and not os.path.exists(timing_dir):
        os.makedirs(timing_dir)
    
    # Run the parallel simulation
    run_simulation_parallel(map_array, people_list, houses_list, buildings_list, 
                          infection_rate=[0.003, 0.00001, 0.000001], 
                          death_rate=[0.01, 0.005, 0.3], 
                          num_days=50, 
                          npy_save_dir=save_dir,
                          timing_output_file=f"{timing_dir}/parallel_{timestamp}.json",
                          infected_people_conscience=0)
    
    # Visualize the initial map with people
    visualize_map(map_array, people_list)
    
    # Print sample of people for debugging
    print("\nSample of people:")
    for i in range(min(5, len(people_list))):
        print(people_list[i])
    
    timestamp = int(time.time())
    save_dir = f'./simulation_data_{timestamp}/'
        
    # Run a full simulation (sequential)
    # if rank == 0:  # Only rank 0 runs the sequential simulation
    #     run_simulation(map_array, people_list, houses_list, buildings_list, 
    #                   infection_rate=[0.003, 0.00001, 0.000001], 
    #                   death_rate=[0.01, 0.005, 0.3], 
    #                   num_days=5, 
    #                   npy_save_dir=save_dir,
    #                   timing_output_file=f"{timing_dir}/sequential_{timestamp}.json")