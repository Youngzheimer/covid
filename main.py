import numpy as np
import matplotlib.pyplot as plt
import random

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

def generate_map(size=100, houses=5, house_size=3, buildings=3, building_size=5, road_width=6):
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

    # 도로를 위치 랜덤이지만 적어도 중앙에서 크게 벗어나진 않도록 세로 가로의 도로 생성
    center = size // 2
    for i in range(int(center + road_width * 2 * random.uniform(0.8, 1.2) - road_width // 2), int(center + road_width * 2 * random.uniform(0.8, 1.2) + road_width // 2)):
        map_array[i, :] = 1  # 수직 도로
        map_array[:, i] = 1  # 수평 도로
    
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
    viz_map = map_array.copy()
    
    # 보기 편하도록 숫자 변경
    # house -> 2
    # building -> 3
    # road -> 1
    # empty space -> 0
    viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
    viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
    
    # Add people to the visualization if provided
    if people_list:
        for person in people_list:
            # Different colors for different infection statuses
            # 5: healthy, 6: infected, 7: recovered, 8: dead
            viz_map[person.x, person.y] = 5 + person.infection_status
    
    plt.figure(figsize=(10, 10))
    
    # Create a custom colormap for better visualization
    import matplotlib.colors as mcolors
    colors = ['white', 'gray', 'green', 'blue', 'black', 'yellow', 'red', 'purple', 'black']
    cmap = mcolors.ListedColormap(colors)
    
    # Set bounds for colorbar
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    plt.imshow(viz_map, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Create custom colorbar
    cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5])
    cbar.ax.set_yticklabels(['Empty', 'Road', 'House', 'Building', 'Healthy', 'Infected', 'Recovered', 'Dead'])
    
    plt.title('COVID-19 Simulation Map')
    plt.show()

def run_simulation(map_array, people_list, houses_list, buildings_list, num_days=100, infection_rate=[0.3, 0.01, 0.001], recovery_rate=[0.1, 0.05, 0.02], death_rate=[0.01, 0.005, 0.02], infected_people_conscience=0.7):
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
    """
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
            # Simulate movement of people
            simulate_movement(people_list, map_array, houses_list, buildings_list)
            # Simulate infection spread
            visualize_map(map_array, people_list)
            
            # Check if all people have reached their destinations
            if all(person.dest_type is None for person in people_list):
                break
        
        print(f"Day {day + 1} everyone has reached their work or school or sum idk.")

        # Simulate infection spread inside houses and buildings


        # And go home yeyeyeyeyeyeyeyeyeyeyeyeyey   
        for person in people_list:
            person.dest_type = 'house'
            person.dest_id = person.house_id  # Go back to their house

        while True:
            # Simulate movement of people
            simulate_movement(people_list, map_array, houses_list, buildings_list)
            # Simulate infection spread
            visualize_map(map_array, people_list)

            # Check if all people have reached their destinations
            if all(person.dest_type is None for person in people_list):
                break
        
        print(f"Day {day + 1} everyone has reached their home.")
        print(f"Day {day + 1} simulation step completed.")

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

        
if __name__ == "__main__":
    # Example usage
    map_size = 100
    map_array, houses_list, buildings_list = generate_map(
        size=map_size, 
        houses=10, 
        house_size=3, 
        buildings=5, 
        building_size=5
    )
    
    # Generate people and place them in houses
    people_list = generate_people(
        map_array, 
        houses_list, 
        person_min_per_house=1, 
        person_max_per_house=1, 
        infection_rate=0.1
    )
    
    print(f"Map size: {map_size}x{map_size}")
    print(f"Houses: {len(houses_list)}")
    print(f"Buildings: {len(buildings_list)}")
    print(f"People: {len(people_list)}")
    
    # Visualize the initial map with people
    visualize_map(map_array, people_list)
    
    # Print sample of people for debugging
    print("\nSample of people:")
    for i in range(min(5, len(people_list))):
        print(people_list[i])
        
    # Uncomment to run a full simulation
    run_simulation(map_array, people_list, houses_list, buildings_list)