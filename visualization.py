import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import platform
import matplotlib.font_manager as fm

# 한글 폰트 설정
# 한글 폰트 설정 (cross-platform)
system = platform.system()
if system == 'Darwin':  # macOS
    plt.rcParams['font.family'] = 'AppleGothic'
elif system == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
else:  # Linux and others
    try:
        # 나눔고딕이 설치되어 있는지 확인
        font_list = fm.findSystemFonts(fontpaths=None, fontext='ttf')
        nanum_fonts = [f for f in font_list if 'NanumGothic' in f]
        if nanum_fonts:
            plt.rcParams['font.family'] = 'NanumGothic'
        else:
            # 대체 폰트들 시도
            possible_fonts = ['NanumGothic', 'UnDotum', 'DejaVu Sans', 'Noto Sans CJK KR']
            for font in possible_fonts:
                try:
                    fm.findfont(font)
                    plt.rcParams['font.family'] = font
                    break
                except:
                    continue
    except:
        pass  # 폰트를 찾지 못하면 기본 폰트 사용
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class SimulationAnalyzer:
    """
    COVID-19 시뮬레이션 데이터를 분석하고 시각화하는 클래스
    """
    def __init__(self, simulation_dir):
        """
        시뮬레이션 디렉토리에서 데이터를 로드합니다.
        
        Parameters:
        - simulation_dir: 시뮬레이션 데이터가 저장된 디렉토리 경로
        """
        self.simulation_dir = simulation_dir
        self.timestamps = []
        self.map_arrays = []
        self.people_data = []
        self.houses_data = []
        self.buildings_data = []
        
        # 데이터 로드
        self._load_simulation_data()
        
        # 색상 매핑 설정
        self.colors = ['white', 'gray', 'green', 'blue', 'black', 'yellow', 'red', 'purple', 'black']
        self.cmap = mcolors.ListedColormap(self.colors)
        self.bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)
        
    def _load_simulation_data(self):
        """
        주어진 디렉토리에서 모든 시뮬레이션 데이터를 로드합니다.
        """
        # 맵 데이터 파일 찾기
        map_files = sorted(glob.glob(os.path.join(self.simulation_dir, '*_map_array.npy')))
        
        for map_file in map_files:
            # 타임스탬프 추출
            timestamp = os.path.basename(map_file).split('_')[0]
            self.timestamps.append(timestamp)
            
            # 맵 배열 로드
            map_array = np.load(map_file)
            self.map_arrays.append(map_array)
            
            # 사람 데이터 로드
            people_file = os.path.join(self.simulation_dir, f'{timestamp}_people.npy')
            if os.path.exists(people_file):
                self.people_data.append(np.load(people_file))
            else:
                print(f"Warning: People file not found for timestamp {timestamp}")
                self.people_data.append(None)
            
            # 집 데이터 로드
            houses_file = os.path.join(self.simulation_dir, f'{timestamp}_houses.npy')
            if os.path.exists(houses_file):
                self.houses_data.append(np.load(houses_file))
            else:
                print(f"Warning: Houses file not found for timestamp {timestamp}")
                self.houses_data.append(None)
            
            # 건물 데이터 로드
            buildings_file = os.path.join(self.simulation_dir, f'{timestamp}_buildings.npy')
            if os.path.exists(buildings_file):
                self.buildings_data.append(np.load(buildings_file))
            else:
                print(f"Warning: Buildings file not found for timestamp {timestamp}")
                self.buildings_data.append(None)
        
        print(f"Loaded {len(self.timestamps)} timestamps from {self.simulation_dir}")
    
    def create_single_map_visualization(self, index=0, show_people=True, figsize=(10, 10)):
        """
        특정 시점의 맵을 시각화합니다.
        
        Parameters:
        - index: 시각화할 타임스탬프 인덱스
        - show_people: 사람을 맵에 표시할지 여부
        - figsize: 그림 크기
        """
        if index < 0 or index >= len(self.timestamps):
            print(f"Error: Index {index} out of range (0-{len(self.timestamps) - 1})")
            return
        
        # 맵 데이터 준비
        viz_map = self.map_arrays[index].copy()
        
        # 맵 코드 변환
        viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
        viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
        
        # 사람 데이터 추가
        if show_people and self.people_data[index] is not None:
            for person in self.people_data[index]:
                # 다른 감염 상태에 대한 다른 색상
                # 5: 건강, 6: 감염, 7: 회복, 8: 사망
                viz_map[person['x'], person['y']] = 5 + person['infection_status']
        
        plt.figure(figsize=figsize)
        
        plt.imshow(viz_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
        
        # 컬러바 생성
        cbar = plt.colorbar(ticks=[0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5])
        cbar.ax.set_yticklabels(['빈 공간', '도로', '집', '건물', '건강', '감염', '회복', '사망'])
        
        plt.title(f'COVID-19 시뮬레이션 맵 (시점: {self.timestamps[index]})')
        plt.show()
    
    def create_infection_timeline_graph(self, figsize=(12, 6)):
        """
        시간에 따른 감염, 회복, 사망자 수의 변화를 그래프로 나타냅니다.
        """
        # 각 상태별 인원 추적
        healthy_counts = []
        infected_counts = []
        recovered_counts = []
        dead_counts = []
        
        for people in self.people_data:
            if people is not None:
                # 각 상태 별 인원 수 계산
                healthy = np.sum(people['infection_status'] == 0)
                infected = np.sum(people['infection_status'] == 1)
                recovered = np.sum(people['infection_status'] == 2)
                dead = np.sum(people['infection_status'] == 3)
                
                healthy_counts.append(healthy)
                infected_counts.append(infected)
                recovered_counts.append(recovered)
                dead_counts.append(dead)
        
        # 타임라인 설정 - 각 날짜의 시작과 끝 부분만 그래프에 표시
        day_markers = []
        day_labels = []
        current_day = -1
        
        for i, timestamp in enumerate(self.timestamps):
            # 시간대별로 일자 계산 (단순화를 위해 5단위로 나눔)
            day = i // 5
            if day != current_day:
                current_day = day
                day_markers.append(i)
                day_labels.append(f"Day {day}")
        
        # 그래프 그리기
        plt.figure(figsize=figsize)
        
        plt.plot(healthy_counts, label='건강', color='yellow')
        plt.plot(infected_counts, label='감염', color='red')
        plt.plot(recovered_counts, label='회복', color='purple')
        plt.plot(dead_counts, label='사망', color='black')
        
        plt.xlabel('시간')
        plt.ylabel('인원 수')
        plt.title('시간에 따른 인구 상태 변화')
        plt.legend()
        
        # 날짜 표시
        plt.xticks(day_markers, day_labels)
        plt.grid(True, alpha=0.3)
        
        plt.show()
        
        # 감염률, 회복률, 사망률 계산
        total_population = np.array(healthy_counts) + np.array(infected_counts) + np.array(recovered_counts) + np.array(dead_counts)
        infection_rate = np.array(infected_counts) / total_population * 100
        recovery_rate = np.array(recovered_counts) / total_population * 100
        mortality_rate = np.array(dead_counts) / total_population * 100
        
        # 비율 그래프 그리기
        plt.figure(figsize=figsize)
        
        plt.plot(infection_rate, label='감염률 (%)', color='red')
        plt.plot(recovery_rate, label='회복률 (%)', color='purple')
        plt.plot(mortality_rate, label='사망률 (%)', color='black')
        
        plt.xlabel('시간')
        plt.ylabel('백분율 (%)')
        plt.title('시간에 따른 인구 상태 비율 변화')
        plt.legend()
        
        # 날짜 표시
        plt.xticks(day_markers, day_labels)
        plt.grid(True, alpha=0.3)
        
        plt.show()
    
    def create_age_group_analysis(self, index=-1):
        """
        연령 그룹별 감염, 회복, 사망률 분석
        
        Parameters:
        - index: 분석할 타임스탬프 인덱스 (-1은 마지막 데이터)
        """
        if index < 0:
            index = len(self.timestamps) + index
        
        if index < 0 or index >= len(self.timestamps):
            print(f"Error: Index {index} out of range (0-{len(self.timestamps) - 1})")
            return
        
        people = self.people_data[index]
        if people is None:
            print("Error: No people data available")
            return
        
        # 연령대별 상태 계산
        age_groups = ['어린이', '성인', '노인']
        statuses = ['건강', '감염', '회복', '사망']
        
        # 연령대 및 상태별 인원 수 저장 배열
        age_status_counts = np.zeros((3, 4), dtype=int)
        
        # 각 사람의 연령대와 상태에 따라 카운트
        for person in people:
            age_group = person['age_group']
            status = person['infection_status']
            age_status_counts[age_group, status] += 1
        
        # 각 연령대별 총 인원
        age_totals = np.sum(age_status_counts, axis=1)
        
        # 그래프를 위한 데이터 준비
        data_for_bars = []
        for i in range(3):  # 연령대
            for j in range(4):  # 상태
                percentage = (age_status_counts[i, j] / age_totals[i] * 100) if age_totals[i] > 0 else 0
                data_for_bars.append({
                    'Age Group': age_groups[i],
                    'Status': statuses[j],
                    'Percentage': percentage
                })
        
        df = pd.DataFrame(data_for_bars)
        
        # 누적 막대 그래프 그리기
        plt.figure(figsize=(10, 6))
        
        ax = sns.barplot(x='Age Group', y='Percentage', hue='Status', data=df)
        
        plt.title('연령대별 감염 상태 분포')
        plt.ylabel('백분율 (%)')
        plt.legend(title='상태')
        
        # 각 막대에 백분율 표시
        for p in ax.patches:
            if p.get_height() > 3:  # 너무 작은 값은 텍스트 표시 안함
                ax.annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom',
                           xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.show()
    
    def create_mask_effectiveness_analysis(self, index=-1):
        """
        마스크 착용 효과 분석
        
        Parameters:
        - index: 분석할 타임스탬프 인덱스 (-1은 마지막 데이터)
        """
        if index < 0:
            index = len(self.timestamps) + index
        
        if index < 0 or index >= len(self.timestamps):
            print(f"Error: Index {index} out of range (0-{len(self.timestamps) - 1})")
            return
        
        people = self.people_data[index]
        if people is None:
            print("Error: No people data available")
            return
        
        # 마스크 종류별 상태 계산
        mask_types = ['마스크 없음', '마스크', '마스크+보호막']
        statuses = ['건강', '감염', '회복', '사망']
        
        # 마스크 종류 및 상태별 인원 수 저장 배열
        mask_status_counts = np.zeros((3, 4), dtype=int)
        
        # 각 사람의 마스크 종류와 상태에 따라 카운트
        for person in people:
            mask_type = person['mask_status']
            status = person['infection_status']
            mask_status_counts[mask_type, status] += 1
        
        # 각 마스크 종류별 총 인원
        mask_totals = np.sum(mask_status_counts, axis=1)
        
        # 그래프를 위한 데이터 준비
        data_for_bars = []
        for i in range(3):  # 마스크 종류
            for j in range(4):  # 상태
                percentage = (mask_status_counts[i, j] / mask_totals[i] * 100) if mask_totals[i] > 0 else 0
                data_for_bars.append({
                    'Mask Type': mask_types[i],
                    'Status': statuses[j],
                    'Percentage': percentage
                })
        
        df = pd.DataFrame(data_for_bars)
        
        # 누적 막대 그래프 그리기
        plt.figure(figsize=(10, 6))
        
        ax = sns.barplot(x='Mask Type', y='Percentage', hue='Status', data=df)
        
        plt.title('마스크 종류별 감염 상태 분포')
        plt.ylabel('백분율 (%)')
        plt.legend(title='상태')
        
        # 각 막대에 백분율 표시
        for p in ax.patches:
            if p.get_height() > 3:  # 너무 작은 값은 텍스트 표시 안함
                ax.annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'bottom',
                           xytext = (0, 5), textcoords = 'offset points')
        
        plt.tight_layout()
        plt.show()
    
    def create_infection_hotspot_map(self, figsize=(12, 10)):
        """
        감염 핫스팟을 히트맵으로 표시합니다.
        시뮬레이션 전체 기간 동안 각 위치에서 발생한 감염의 총 수를 보여줍니다.
        """
        # 맵 크기 가져오기
        map_size = self.map_arrays[0].shape[0]
        
        # 각 위치의 감염 발생 횟수 추적
        infection_counts = np.zeros((map_size, map_size), dtype=int)
        
        # 마지막 맵으로 기본 구조 설정
        last_map = self.map_arrays[-1].copy()
        
        # 건물과 집 표시
        buildings_houses_map = np.zeros_like(last_map)
        buildings_houses_map[(last_map >= 200) & (last_map <= 299)] = 1  # 집
        buildings_houses_map[(last_map >= 300) & (last_map <= 399)] = 2  # 건물
        
        # 타임스탬프별로 감염자 위치 추적
        for people in self.people_data:
            if people is not None:
                for person in people:
                    if person['infection_status'] == 1:  # 감염 상태
                        x, y = person['x'], person['y']
                        if 0 <= x < map_size and 0 <= y < map_size:
                            infection_counts[x, y] += 1
        
        # 감염 히트맵과 건물/집 맵 시각화
        plt.figure(figsize=figsize)
        
        # 1. 감염 히트맵
        plt.subplot(1, 2, 1)
        plt.imshow(infection_counts.T, cmap='hot', interpolation='nearest')
        plt.colorbar(label='감염 발생 수')
        plt.title('감염 핫스팟 맵')
        
        # 2. 건물/집 위치와 함께 감염 히트맵
        plt.subplot(1, 2, 2)
        
        # 기본 맵 표시 (건물/집)
        buildings_houses_cmap = mcolors.ListedColormap(['white', 'green', 'blue'])
        plt.imshow(buildings_houses_map.T, cmap=buildings_houses_cmap, alpha=0.5)
        
        # 감염 히트맵 오버레이
        plt.imshow(infection_counts.T, cmap='hot', alpha=0.5, interpolation='nearest')
        
        plt.colorbar(label='감염 발생 수')
        plt.title('건물/집 위치와 함께 보는 감염 핫스팟 맵')
        
        plt.tight_layout()
        plt.show()
    
    def create_movement_density_map(self, figsize=(12, 10)):
        """
        사람들의 이동 밀도를 히트맵으로 표시합니다.
        각 위치에 사람이 있었던 총 횟수를 표시합니다.
        """
        # 맵 크기 가져오기
        map_size = self.map_arrays[0].shape[0]
        
        # 각 위치의 사람 발생 횟수 추적
        movement_counts = np.zeros((map_size, map_size), dtype=int)
        
        # 마지막 맵으로 기본 구조 설정
        last_map = self.map_arrays[-1].copy()
        
        # 건물과 집 표시
        buildings_houses_map = np.zeros_like(last_map)
        buildings_houses_map[(last_map >= 200) & (last_map <= 299)] = 1  # 집
        buildings_houses_map[(last_map >= 300) & (last_map <= 399)] = 2  # 건물
        roads_map = (last_map == 1).astype(int)  # 도로
        
        # 타임스탬프별로 모든 사람의 위치 추적
        for people in self.people_data:
            if people is not None:
                for person in people:
                    x, y = person['x'], person['y']
                    if 0 <= x < map_size and 0 <= y < map_size:
                        movement_counts[x, y] += 1
        
        # 이동 밀도 히트맵 시각화
        plt.figure(figsize=figsize)
        
        # 1. 이동 밀도 히트맵
        plt.subplot(1, 2, 1)
        plt.imshow(movement_counts.T, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='사람 발생 수')
        plt.title('이동 밀도 맵')
        
        # 2. 도로/건물/집 위치와 함께 이동 밀도 히트맵
        plt.subplot(1, 2, 2)
        
        # 도로 맵 표시
        plt.imshow(roads_map.T, cmap='binary', alpha=0.3)
        
        # 건물/집 표시
        buildings_houses_cmap = mcolors.ListedColormap(['white', 'green', 'blue'])
        plt.imshow(buildings_houses_map.T, cmap=buildings_houses_cmap, alpha=0.3)
        
        # 이동 밀도 히트맵 오버레이
        plt.imshow(movement_counts.T, cmap='viridis', alpha=0.7, interpolation='nearest')
        
        plt.colorbar(label='사람 발생 수')
        plt.title('도로/건물/집 위치와 함께 보는 이동 밀도 맵')
        
        plt.tight_layout()
        plt.show()
    
    def create_3d_infection_spread_animation(self, frames=10, interval=200, figsize=(12, 8)):
        """
        시간에 따른 3D 감염 확산 애니메이션을 생성합니다.
        
        Parameters:
        - frames: 애니메이션에 사용할 프레임(타임스탬프) 수
        - interval: 프레임 간 밀리초 단위 간격
        """
        # 사용할 프레임 선택
        if frames > len(self.timestamps):
            frames = len(self.timestamps)
        
        frame_indices = np.linspace(0, len(self.timestamps)-1, frames, dtype=int)
        
        # 맵 크기 가져오기
        map_size = self.map_arrays[0].shape[0]
        
        # 3D 그림 설정
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 애니메이션을 위한 스캐터 플롯
        scatter = ax.scatter([], [], [], c=[], cmap='viridis', s=10)
        
        # 제목 텍스트 객체
        title = ax.set_title('')
        
        # 건물과 도로의 기본 위치
        roads = []
        houses = []
        buildings = []
        
        # 도로, 집, 건물 위치 추출
        map_array = self.map_arrays[0]
        
        # 도로 위치
        road_points = np.argwhere(map_array == 1)
        for x, y in road_points:
            roads.append((x, y, 0))  # z=0 (바닥)
        
        # 집 위치
        houses_array = self.houses_data[0]
        if houses_array is not None:
            for house in houses_array:
                x, y, size = house['x'], house['y'], house['size']
                houses.append((x, y, size))
        
        # 건물 위치
        buildings_array = self.buildings_data[0]
        if buildings_array is not None:
            for building in buildings_array:
                x, y, size = building['x'], building['y'], building['size']
                buildings.append((x, y, size))
        
        # 도로, 집, 건물 시각화 (고정)
        roads = np.array(roads)
        if len(roads) > 0:
            ax.scatter(roads[:, 0], roads[:, 1], roads[:, 2], c='gray', alpha=0.2, s=5)
        
        # 집 시각화 (상자로)
        for x, y, size in houses:
            ax.add_collection3d(plt.matplotlib.collections.PolyCollection(
                [np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])],
                closed=True, facecolor='green', alpha=0.3
            ))
        
        # 건물 시각화 (상자로)
        for x, y, size in buildings:
            ax.add_collection3d(plt.matplotlib.collections.PolyCollection(
                [np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])],
                closed=True, facecolor='blue', alpha=0.3
            ))
        
        def update(frame_idx):
            """애니메이션 프레임 업데이트 함수"""
            ax.clear()
            
            idx = frame_indices[frame_idx]
            people = self.people_data[idx]
            
            if people is None:
                return scatter,
            
            # 사람들의 위치 및 상태 추출
            xs = []
            ys = []
            zs = []
            colors = []
            
            for person in people:
                if person['infection_status'] != 3:  # 사망자 제외
                    xs.append(person['x'])
                    ys.append(person['y'])
                    # z 좌표는 감염 상태에 따라 다르게 설정
                    if person['infection_status'] == 0:  # 건강
                        zs.append(1)
                        colors.append('yellow')
                    elif person['infection_status'] == 1:  # 감염
                        zs.append(2)
                        colors.append('red')
                    else:  # 회복
                        zs.append(1.5)
                        colors.append('purple')
            
            # 도로 시각화
            if len(roads) > 0:
                ax.scatter(roads[:, 0], roads[:, 1], roads[:, 2], c='gray', alpha=0.2, s=5)
            
            # 사람들 시각화
            scatter = ax.scatter(xs, ys, zs, c=colors, s=10)
            
            # 집 시각화
            for x, y, size in houses:
                ax.add_collection3d(plt.matplotlib.collections.PolyCollection(
                    [np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])],
                    closed=True, facecolor='green', alpha=0.3
                ))
            
            # 건물 시각화
            for x, y, size in buildings:
                ax.add_collection3d(plt.matplotlib.collections.PolyCollection(
                    [np.array([[x, y], [x+size, y], [x+size, y+size], [x, y+size]])],
                    closed=True, facecolor='blue', alpha=0.3
                ))
            
            ax.set_xlim(0, map_size)
            ax.set_ylim(0, map_size)
            ax.set_zlim(0, 3)
            
            ax.set_title(f'시간: {self.timestamps[idx]}')
            
            return scatter,
        
        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)
        
        plt.show()
        
        return ani
    
    def create_map_animation(self, frames=10, interval=200, figsize=(10, 10)):
        """
        시간에 따른 맵 변화를 애니메이션으로 보여줍니다.
        
        Parameters:
        - frames: 애니메이션에 사용할 프레임(타임스탬프) 수
        - interval: 프레임 간 밀리초 단위 간격
        """
        # 사용할 프레임 선택
        if frames > len(self.timestamps):
            frames = len(self.timestamps)
        
        frame_indices = np.linspace(0, len(self.timestamps)-1, frames, dtype=int)
        
        # 그림 설정
        fig, ax = plt.subplots(figsize=figsize)
        
        def update(frame_idx):
            """애니메이션 프레임 업데이트 함수"""
            ax.clear()
            
            idx = frame_indices[frame_idx]
            map_array = self.map_arrays[idx]
            people = self.people_data[idx]
            
            # 맵 데이터 준비
            viz_map = map_array.copy()
            
            # 맵 코드 변환
            viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
            viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
            
            # 사람 데이터 추가
            if people is not None:
                for person in people:
                    # 다른 감염 상태에 대한 다른 색상
                    viz_map[person['x'], person['y']] = 5 + person['infection_status']
            
            # 맵 표시
            im = ax.imshow(viz_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            
            # 제목 업데이트
            ax.set_title(f'시간: {self.timestamps[idx]}')
            
            return [im]
        
        # 컬러바 추가
        cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm), cax=cax)
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5])
        cbar.ax.set_yticklabels(['빈 공간', '도로', '집', '건물', '건강', '감염', '회복', '사망'])
        
        ani = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def create_statistics_dashboard(self, index=-1):
        """
        주요 통계 정보를 대시보드 형태로 표시합니다.
        
        Parameters:
        - index: 분석할 타임스탬프 인덱스 (-1은 마지막 데이터)
        """
        if index < 0:
            index = len(self.timestamps) + index
        
        if index < 0 or index >= len(self.timestamps):
            print(f"Error: Index {index} out of range (0-{len(self.timestamps) - 1})")
            return
        
        people = self.people_data[index]
        if people is None:
            print("Error: No people data available")
            return
        
        # 기본 통계 정보 계산
        total_people = len(people)
        healthy_count = np.sum(people['infection_status'] == 0)
        infected_count = np.sum(people['infection_status'] == 1)
        recovered_count = np.sum(people['infection_status'] == 2)
        dead_count = np.sum(people['infection_status'] == 3)
        
        # 연령대별 통계
        children_count = np.sum(people['age_group'] == 0)
        adult_count = np.sum(people['age_group'] == 1)
        senior_count = np.sum(people['age_group'] == 2)
        
        # 마스크 착용별 통계
        no_mask_count = np.sum(people['mask_status'] == 0)
        mask_count = np.sum(people['mask_status'] == 1)
        mask_shield_count = np.sum(people['mask_status'] == 2)
        
        # 마스크 착용별 감염률
        no_mask_infected = np.sum((people['mask_status'] == 0) & ((people['infection_status'] == 1) | (people['infection_status'] == 2)))
        mask_infected = np.sum((people['mask_status'] == 1) & ((people['infection_status'] == 1) | (people['infection_status'] == 2)))
        mask_shield_infected = np.sum((people['mask_status'] == 2) & ((people['infection_status'] == 1) | (people['infection_status'] == 2)))
        
        no_mask_infection_rate = (no_mask_infected / no_mask_count * 100) if no_mask_count > 0 else 0
        mask_infection_rate = (mask_infected / mask_count * 100) if mask_count > 0 else 0
        mask_shield_infection_rate = (mask_shield_infected / mask_shield_count * 100) if mask_shield_count > 0 else 0
        
        # 대시보드 생성
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 인구 상태 파이 차트
        ax1 = fig.add_subplot(2, 3, 1)
        status_labels = ['건강', '감염', '회복', '사망']
        status_counts = [healthy_count, infected_count, recovered_count, dead_count]
        status_colors = ['yellow', 'red', 'purple', 'black']
        ax1.pie(status_counts, labels=status_labels, colors=status_colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('인구 상태 분포')
        
        # 2. 연령대별 분포 파이 차트
        ax2 = fig.add_subplot(2, 3, 2)
        age_labels = ['어린이', '성인', '노인']
        age_counts = [children_count, adult_count, senior_count]
        ax2.pie(age_counts, labels=age_labels, autopct='%1.1f%%', startangle=90)
        ax2.set_title('연령대별 분포')
        
        # 3. 마스크 착용 분포 파이 차트
        ax3 = fig.add_subplot(2, 3, 3)
        mask_labels = ['마스크 없음', '마스크', '마스크+보호막']
        mask_counts = [no_mask_count, mask_count, mask_shield_count]
        ax3.pie(mask_counts, labels=mask_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('마스크 착용 분포')
        
        # 4. 마스크 종류별 감염률 막대 그래프
        ax4 = fig.add_subplot(2, 3, 4)
        mask_types = ['마스크 없음', '마스크', '마스크+보호막']
        infection_rates = [no_mask_infection_rate, mask_infection_rate, mask_shield_infection_rate]
        ax4.bar(mask_types, infection_rates, color=['red', 'orange', 'green'])
        ax4.set_title('마스크 종류별 감염률')
        ax4.set_ylabel('감염률 (%)')
        
        # 5. 연령대별 감염 상태 막대 그래프
        ax5 = fig.add_subplot(2, 3, 5)
        
        # 연령대별 상태 계산
        children_status = [
            np.sum((people['age_group'] == 0) & (people['infection_status'] == 0)),
            np.sum((people['age_group'] == 0) & (people['infection_status'] == 1)),
            np.sum((people['age_group'] == 0) & (people['infection_status'] == 2)),
            np.sum((people['age_group'] == 0) & (people['infection_status'] == 3))
        ]
        
        adult_status = [
            np.sum((people['age_group'] == 1) & (people['infection_status'] == 0)),
            np.sum((people['age_group'] == 1) & (people['infection_status'] == 1)),
            np.sum((people['age_group'] == 1) & (people['infection_status'] == 2)),
            np.sum((people['age_group'] == 1) & (people['infection_status'] == 3))
        ]
        
        senior_status = [
            np.sum((people['age_group'] == 2) & (people['infection_status'] == 0)),
            np.sum((people['age_group'] == 2) & (people['infection_status'] == 1)),
            np.sum((people['age_group'] == 2) & (people['infection_status'] == 2)),
            np.sum((people['age_group'] == 2) & (people['infection_status'] == 3))
        ]
        
        # 퍼센트로 변환
        children_percent = [count / children_count * 100 if children_count > 0 else 0 for count in children_status]
        adult_percent = [count / adult_count * 100 if adult_count > 0 else 0 for count in adult_status]
        senior_percent = [count / senior_count * 100 if senior_count > 0 else 0 for count in senior_status]
        
        x = np.arange(len(status_labels))
        width = 0.25
        
        ax5.bar(x - width, children_percent, width, label='어린이', color='lightblue')
        ax5.bar(x, adult_percent, width, label='성인', color='lightgreen')
        ax5.bar(x + width, senior_percent, width, label='노인', color='salmon')
        
        ax5.set_title('연령대별 감염 상태 (%)')
        ax5.set_xticks(x)
        ax5.set_xticklabels(status_labels)
        ax5.legend()
        
        # 6. 텍스트 정보
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        info_text = f"""
        시뮬레이션 정보:
        
        시간: {self.timestamps[index]}
        
        총 인구: {total_people}
        
        건강한 사람: {healthy_count} ({healthy_count/total_people*100:.1f}%)
        감염된 사람: {infected_count} ({infected_count/total_people*100:.1f}%)
        회복된 사람: {recovered_count} ({recovered_count/total_people*100:.1f}%)
        사망자: {dead_count} ({dead_count/total_people*100:.1f}%)
        
        감염률(현재+회복+사망): {(infected_count + recovered_count + dead_count)/total_people*100:.1f}%
        사망률(전체 중): {dead_count/total_people*100:.1f}%
        사망률(감염자 중): {dead_count/(infected_count + recovered_count + dead_count)*100:.1f}% 
        """
        ax6.text(0, 0.5, info_text, fontsize=10)
        
        plt.tight_layout()
        plt.suptitle(f'COVID-19 시뮬레이션 통계 대시보드 (시점: {self.timestamps[index]})', fontsize=16, y=1.02)
        plt.show()
    
    def save_all_visualizations(self, output_dir, save_format='png'):
        """
        모든 시각화를 파일로 저장합니다.
        
        Parameters:
        - output_dir: 결과 저장할 디렉토리
        - save_format: 저장 형식 ('png', 'pdf', 'jpg' 등)
        """
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        print(f"Saving visualizations to {output_dir}...")
        
        # 1. 첫 번째 맵 저장
        plt.figure(figsize=(10, 10))
        self.create_single_map_visualization(index=0, show_people=True)
        plt.savefig(os.path.join(output_dir, f'initial_map.{save_format}'), format=save_format)
        plt.close()
        
        # 2. 마지막 맵 저장
        plt.figure(figsize=(10, 10))
        self.create_single_map_visualization(index=-1, show_people=True)
        plt.savefig(os.path.join(output_dir, f'final_map.{save_format}'), format=save_format)
        plt.close()
        
        # 3. 감염 타임라인 그래프 저장
        self.create_infection_timeline_graph()
        plt.savefig(os.path.join(output_dir, f'infection_timeline.{save_format}'), format=save_format)
        plt.close()
        
        # 4. 연령대별 분석 저장
        self.create_age_group_analysis()
        plt.savefig(os.path.join(output_dir, f'age_group_analysis.{save_format}'), format=save_format)
        plt.close()
        
        # 5. 마스크 효과 분석 저장
        self.create_mask_effectiveness_analysis()
        plt.savefig(os.path.join(output_dir, f'mask_effectiveness.{save_format}'), format=save_format)
        plt.close()
        
        # 6. 감염 핫스팟 맵 저장
        self.create_infection_hotspot_map()
        plt.savefig(os.path.join(output_dir, f'infection_hotspot_map.{save_format}'), format=save_format)
        plt.close()
        
        # 7. 이동 밀도 맵 저장
        self.create_movement_density_map()
        plt.savefig(os.path.join(output_dir, f'movement_density_map.{save_format}'), format=save_format)
        plt.close()
        
        # 8. 통계 대시보드 저장
        self.create_statistics_dashboard()
        plt.savefig(os.path.join(output_dir, f'statistics_dashboard.{save_format}'), format=save_format)
        plt.close()
        
        print(f"All visualizations have been saved to {output_dir}")


# 사용 예제
if __name__ == "__main__":
    # 가장 최근 시뮬레이션 디렉토리 찾기
    simulation_dirs = glob.glob('./simulation_data_*/')
    
    if not simulation_dirs:
        print("No simulation directories found. Please run the simulation first.")
        exit(1)
    
    # 가장 최근 디렉토리 선택
    latest_dir = max(simulation_dirs, key=os.path.getctime)
    print(f"Using latest simulation directory: {latest_dir}")
    
    # 시뮬레이션 분석기 초기화
    analyzer = SimulationAnalyzer(latest_dir)
    
    # 초기 맵 시각화
    print("Creating initial map visualization...")
    analyzer.create_single_map_visualization(index=0, show_people=True)
    
    # 마지막 맵 시각화
    print("Creating final map visualization...")
    analyzer.create_single_map_visualization(index=-1, show_people=True)
    
    # 감염 타임라인 그래프
    print("Creating infection timeline graph...")
    analyzer.create_infection_timeline_graph()
    
    # 연령대별 분석
    print("Creating age group analysis...")
    analyzer.create_age_group_analysis()
    
    # 마스크 효과 분석
    print("Creating mask effectiveness analysis...")
    analyzer.create_mask_effectiveness_analysis()
    
    # 감염 핫스팟 맵
    print("Creating infection hotspot map...")
    analyzer.create_infection_hotspot_map()
    
    # 이동 밀도 맵
    print("Creating movement density map...")
    analyzer.create_movement_density_map()
    
    # 통계 대시보드
    print("Creating statistics dashboard...")
    analyzer.create_statistics_dashboard()
    
    # 맵 애니메이션 (프레임 수를 줄여서 더 빠르게 실행)
    print("Creating map animation...")
    analyzer.create_map_animation(frames=20)
    
    # # 3D 시각화 (선택 사항, 시간이 오래 걸릴 수 있음)
    # print("Creating 3D animation...")
    # analyzer.create_3d_infection_spread_animation(frames=10)
    
    # # 모든 시각화 저장 (선택 사항)
    # print("Saving all visualizations...")
    # output_dir = f"./viz_results_{int(time.time())}/"
    # analyzer.save_all_visualizations(output_dir)
