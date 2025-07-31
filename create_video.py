import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import glob
import cv2
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # 화면 출력 없이 이미지 저장을 위한 백엔드 설정
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

class VideoCreator:
    """
    COVID-19 시뮬레이션 데이터를 기반으로 영상을 생성하는 클래스
    """
    def __init__(self, simulation_dir, output_dir='./videos'):
        """
        초기화 함수
        
        Parameters:
        - simulation_dir: 시뮬레이션 데이터가 저장된 디렉토리 경로
        - output_dir: 영상이 저장될 디렉토리 경로
        """
        self.simulation_dir = simulation_dir
        self.output_dir = output_dir
        self.timestamps = []
        self.map_arrays = []
        self.people_data = []
        self.houses_data = []
        self.buildings_data = []
        
        # 출력 디렉토리가 없으면 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
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

    def create_frames(self, temp_dir='./temp_frames', figsize=(10, 10), dpi=100):
        """
        시뮬레이션의 각 시점을 이미지 프레임으로 저장합니다.
        
        Parameters:
        - temp_dir: 임시 프레임 이미지를 저장할 디렉토리 경로
        - figsize: 그림 크기
        - dpi: 이미지 해상도
        
        Returns:
        - 생성된 프레임 파일 경로 리스트
        """
        # 임시 디렉토리 생성
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            # 기존 이미지 파일 삭제
            for file in glob.glob(os.path.join(temp_dir, '*.png')):
                os.remove(file)
        
        frame_paths = []
        
        print("Creating video frames...")
        for i, timestamp in enumerate(tqdm(self.timestamps)):
            # 프레임 이미지 파일 경로
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            
            # 맵 데이터 준비
            viz_map = self.map_arrays[i].copy()
            
            # 맵 코드 변환
            viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
            viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
            
            # 사람 데이터 추가
            if self.people_data[i] is not None:
                for person in self.people_data[i]:
                    # 다른 감염 상태에 대한 다른 색상
                    # 5: 건강, 6: 감염, 7: 회복, 8: 사망
                    viz_map[person['x'], person['y']] = 5 + person['infection_status']
            
            # 그림 생성
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            im = ax.imshow(viz_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            
            # 컬러바 생성
            cbar = plt.colorbar(im, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5])
            cbar.ax.set_yticklabels(['빈 공간', '도로', '집', '건물', '건강', '감염', '회복', '사망'])
            
            # 제목 설정
            ax.set_title(f'COVID-19 시뮬레이션 맵 (시점: {timestamp})')
            
            # 이미지로 저장
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            
            frame_paths.append(frame_path)
        
        return frame_paths

    def create_infection_stats_frames(self, temp_dir='./temp_stats_frames', figsize=(12, 6), dpi=100):
        """
        감염 통계 그래프의 각 시점을 이미지 프레임으로 저장합니다.
        
        Parameters:
        - temp_dir: 임시 프레임 이미지를 저장할 디렉토리 경로
        - figsize: 그림 크기
        - dpi: 이미지 해상도
        
        Returns:
        - 생성된 프레임 파일 경로 리스트
        """
        # 임시 디렉토리 생성
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            # 기존 이미지 파일 삭제
            for file in glob.glob(os.path.join(temp_dir, '*.png')):
                os.remove(file)
        
        frame_paths = []
        
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
        
        print("Creating statistics frames...")
        for i in tqdm(range(len(self.timestamps))):
            # 프레임 이미지 파일 경로
            frame_path = os.path.join(temp_dir, f'stats_frame_{i:04d}.png')
            
            # 그래프 그리기 (시간에 따른 변화 - i까지의 데이터만 표시)
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            
            ax.plot(healthy_counts[:i+1], label='건강', color='yellow')
            ax.plot(infected_counts[:i+1], label='감염', color='red')
            ax.plot(recovered_counts[:i+1], label='회복', color='purple')
            ax.plot(dead_counts[:i+1], label='사망', color='black')
            
            ax.set_xlabel('시간')
            ax.set_ylabel('인원 수')
            ax.set_title(f'시간에 따른 인구 상태 변화 (현재: {self.timestamps[i]})')
            ax.legend()
            
            # x축 제한 - 일관된 크기 유지
            ax.set_xlim(0, len(self.timestamps))
            
            # y축 제한 - 최대값 기준
            max_count = max(max(healthy_counts), max(infected_counts), max(recovered_counts), max(dead_counts))
            ax.set_ylim(0, max_count * 1.1)  # 여유 공간 10%
            
            # 날짜 표시 (표시 가능한 날짜만)
            visible_markers = [marker for marker in day_markers if marker <= i]
            visible_labels = day_labels[:len(visible_markers)]
            ax.set_xticks(visible_markers)
            ax.set_xticklabels(visible_labels)
            
            ax.grid(True, alpha=0.3)
            
            # 이미지로 저장
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            
            frame_paths.append(frame_path)
        
        return frame_paths
    
    def create_video(self, frame_rate=5, delete_frames=True):
        """
        이미지 프레임을 비디오로 변환합니다.
        
        Parameters:
        - frame_rate: 초당 프레임 수
        - delete_frames: 비디오 생성 후 프레임 이미지 삭제 여부
        
        Returns:
        - 생성된 비디오 파일 경로
        """
        # 맵 프레임 생성
        map_frames = self.create_frames()
        
        # 비디오 파일 경로
        video_path = os.path.join(self.output_dir, f'covid_simulation_{os.path.basename(self.simulation_dir)}.mp4')
        
        # 비디오 생성을 위한 설정
        frame = cv2.imread(map_frames[0])
        height, width, layers = frame.shape
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        
        print("Creating video...")
        for frame_path in tqdm(map_frames):
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        # 비디오 저장
        video.release()
        
        print(f"Video saved to {video_path}")
        
        # 임시 프레임 삭제
        if delete_frames:
            for frame_path in map_frames:
                os.remove(frame_path)
            os.rmdir(os.path.dirname(map_frames[0]))
            print("Temporary frames deleted")
        
        return video_path
    
    def create_stats_video(self, frame_rate=5, delete_frames=True):
        """
        감염 통계 그래프 이미지 프레임을 비디오로 변환합니다.
        
        Parameters:
        - frame_rate: 초당 프레임 수
        - delete_frames: 비디오 생성 후 프레임 이미지 삭제 여부
        
        Returns:
        - 생성된 비디오 파일 경로
        """
        # 통계 그래프 프레임 생성
        stats_frames = self.create_infection_stats_frames()
        
        # 비디오 파일 경로
        video_path = os.path.join(self.output_dir, f'covid_stats_{os.path.basename(self.simulation_dir)}.mp4')
        
        # 비디오 생성을 위한 설정
        frame = cv2.imread(stats_frames[0])
        height, width, layers = frame.shape
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        
        print("Creating statistics video...")
        for frame_path in tqdm(stats_frames):
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        # 비디오 저장
        video.release()
        
        print(f"Statistics video saved to {video_path}")
        
        # 임시 프레임 삭제
        if delete_frames:
            for frame_path in stats_frames:
                os.remove(frame_path)
            os.rmdir(os.path.dirname(stats_frames[0]))
            print("Temporary frames deleted")
        
        return video_path
    
    def create_side_by_side_video(self, frame_rate=5, delete_frames=True):
        """
        맵과 통계 그래프를 나란히 보여주는 비디오를 생성합니다.
        
        Parameters:
        - frame_rate: 초당 프레임 수
        - delete_frames: 비디오 생성 후 프레임 이미지 삭제 여부
        
        Returns:
        - 생성된 비디오 파일 경로
        """
        # 임시 디렉토리 생성
        temp_dir = './temp_combined_frames'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            # 기존 이미지 파일 삭제
            for file in glob.glob(os.path.join(temp_dir, '*.png')):
                os.remove(file)
        
        # 맵 프레임과 통계 프레임 생성
        map_frames = self.create_frames()
        stats_frames = self.create_infection_stats_frames()
        
        # 비디오 파일 경로
        video_path = os.path.join(self.output_dir, f'covid_combined_{os.path.basename(self.simulation_dir)}.mp4')
        
        combined_frames = []
        
        print("Creating combined frames...")
        for i in tqdm(range(len(self.timestamps))):
            # 결합 이미지 파일 경로
            combined_frame_path = os.path.join(temp_dir, f'combined_frame_{i:04d}.png')
            
            # 두 이미지 로드
            map_img = cv2.imread(map_frames[i])
            stats_img = cv2.imread(stats_frames[i])
            
            # 이미지 높이 일치시키기
            map_height, map_width = map_img.shape[:2]
            stats_height, stats_width = stats_img.shape[:2]
            
            # 더 큰 높이에 맞추기
            target_height = max(map_height, stats_height)
            
            # 맵 이미지 리사이즈
            map_new_width = int(map_width * (target_height / map_height))
            map_resized = cv2.resize(map_img, (map_new_width, target_height))
            
            # 통계 이미지 리사이즈
            stats_new_width = int(stats_width * (target_height / stats_height))
            stats_resized = cv2.resize(stats_img, (stats_new_width, target_height))
            
            # 이미지 가로로 합치기
            combined_img = np.hstack((map_resized, stats_resized))
            
            # 저장
            cv2.imwrite(combined_frame_path, combined_img)
            combined_frames.append(combined_frame_path)
        
        # 비디오 생성
        first_frame = cv2.imread(combined_frames[0])
        height, width, layers = first_frame.shape
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        
        print("Creating combined video...")
        for frame_path in tqdm(combined_frames):
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        # 비디오 저장
        video.release()
        
        print(f"Combined video saved to {video_path}")
        
        # 임시 프레임 삭제
        if delete_frames:
            for frame_path in map_frames + stats_frames + combined_frames:
                try:
                    os.remove(frame_path)
                except:
                    pass  # 이미 삭제된 파일 무시
            
            try:
                os.rmdir(os.path.dirname(map_frames[0]))
                os.rmdir(os.path.dirname(stats_frames[0]))
                os.rmdir(temp_dir)
            except:
                pass
                
            print("Temporary frames deleted")
        
        return video_path
    
    def create_dashboard_video(self, frame_rate=5, figsize=(16, 9), dpi=100, delete_frames=True):
        """
        맵, 통계 그래프, 감염 상태 등을 포함한 대시보드 비디오를 생성합니다.
        
        Parameters:
        - frame_rate: 초당 프레임 수
        - figsize: 그림 크기
        - dpi: 이미지 해상도
        - delete_frames: 비디오 생성 후 프레임 이미지 삭제 여부
        
        Returns:
        - 생성된 비디오 파일 경로
        """
        # 임시 디렉토리 생성
        temp_dir = './temp_dashboard_frames'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            # 기존 이미지 파일 삭제
            for file in glob.glob(os.path.join(temp_dir, '*.png')):
                os.remove(file)
        
        dashboard_frames = []
        
        # 상태별 인원 추적 데이터 계산
        healthy_counts = []
        infected_counts = []
        recovered_counts = []
        dead_counts = []
        
        for people in self.people_data:
            if people is not None:
                healthy = np.sum(people['infection_status'] == 0)
                infected = np.sum(people['infection_status'] == 1)
                recovered = np.sum(people['infection_status'] == 2)
                dead = np.sum(people['infection_status'] == 3)
                
                healthy_counts.append(healthy)
                infected_counts.append(infected)
                recovered_counts.append(recovered)
                dead_counts.append(dead)
        
        print("Creating dashboard frames...")
        for i in tqdm(range(len(self.timestamps))):
            # 프레임 이미지 파일 경로
            frame_path = os.path.join(temp_dir, f'dashboard_frame_{i:04d}.png')
            
            # 대시보드 그리기
            fig = plt.figure(figsize=figsize, dpi=dpi)
            
            # 1. 맵 그리기 (왼쪽 상단)
            ax1 = fig.add_subplot(2, 2, 1)
            viz_map = self.map_arrays[i].copy()
            viz_map[(viz_map >= 200) & (viz_map <= 299)] = 2  # 집
            viz_map[(viz_map >= 300) & (viz_map <= 399)] = 3  # 건물
            
            if self.people_data[i] is not None:
                for person in self.people_data[i]:
                    viz_map[person['x'], person['y']] = 5 + person['infection_status']
            
            im = ax1.imshow(viz_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
            ax1.set_title(f'COVID-19 시뮬레이션 맵')
            
            # 컬러바 추가
            cbar = plt.colorbar(im, ax=ax1, ticks=[0.5, 1.5, 2.5, 3.5, 5.5, 6.5, 7.5, 8.5], fraction=0.046, pad=0.04)
            cbar.ax.set_yticklabels(['빈 공간', '도로', '집', '건물', '건강', '감염', '회복', '사망'], fontsize=8)
            
            # 2. 감염 추이 그래프 (오른쪽 상단)
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(healthy_counts[:i+1], label='건강', color='yellow')
            ax2.plot(infected_counts[:i+1], label='감염', color='red')
            ax2.plot(recovered_counts[:i+1], label='회복', color='purple')
            ax2.plot(dead_counts[:i+1], label='사망', color='black')
            
            ax2.set_xlabel('시간')
            ax2.set_ylabel('인원 수')
            ax2.set_title('시간에 따른 인구 상태 변화')
            ax2.legend()
            
            # x축 제한 - 일관된 크기 유지
            ax2.set_xlim(0, len(self.timestamps))
            
            # y축 제한 - 최대값 기준
            max_count = max(max(healthy_counts), max(infected_counts), max(recovered_counts), max(dead_counts))
            ax2.set_ylim(0, max_count * 1.1)  # 여유 공간 10%
            
            ax2.grid(True, alpha=0.3)
            
            # 3. 현재 상태 파이 차트 (왼쪽 하단)
            ax3 = fig.add_subplot(2, 2, 3)
            if self.people_data[i] is not None:
                people = self.people_data[i]
                
                # 상태별 인원 계산
                healthy = np.sum(people['infection_status'] == 0)
                infected = np.sum(people['infection_status'] == 1)
                recovered = np.sum(people['infection_status'] == 2)
                dead = np.sum(people['infection_status'] == 3)
                
                status_labels = ['건강', '감염', '회복', '사망']
                status_counts = [healthy, infected, recovered, dead]
                status_colors = ['yellow', 'red', 'purple', 'black']
                
                ax3.pie(status_counts, labels=status_labels, colors=status_colors, 
                        autopct='%1.1f%%', startangle=90)
                ax3.set_title('현재 인구 상태 분포')
            
            # 4. 정보 텍스트 (오른쪽 하단)
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            
            if self.people_data[i] is not None:
                people = self.people_data[i]
                
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
                
                info_text = f"""
                시뮬레이션 정보:
                
                시간: {self.timestamps[i]}
                
                총 인구: {total_people}
                
                건강한 사람: {healthy_count} ({healthy_count/total_people*100:.1f}%)
                감염된 사람: {infected_count} ({infected_count/total_people*100:.1f}%)
                회복된 사람: {recovered_count} ({recovered_count/total_people*100:.1f}%)
                사망자: {dead_count} ({dead_count/total_people*100:.1f}%)
                
                감염률(현재+회복+사망): {(infected_count + recovered_count + dead_count)/total_people*100:.1f}%
                사망률(전체 중): {dead_count/total_people*100:.1f}%
                사망률(감염자 중): {dead_count/(infected_count + recovered_count + dead_count)*100:.1f}% 
                
                연령 분포: 
                어린이 {children_count}명 ({children_count/total_people*100:.1f}%)
                성인 {adult_count}명 ({adult_count/total_people*100:.1f}%)
                노인 {senior_count}명 ({senior_count/total_people*100:.1f}%)
                """
                ax4.text(0, 0.5, info_text, fontsize=10)
            
            plt.tight_layout()
            plt.suptitle(f'COVID-19 시뮬레이션 대시보드 (시점: {self.timestamps[i]})', fontsize=16, y=0.98)
            
            # 이미지로 저장
            plt.savefig(frame_path, bbox_inches='tight')
            plt.close(fig)
            
            dashboard_frames.append(frame_path)
        
        # 비디오 파일 경로
        video_path = os.path.join(self.output_dir, f'covid_dashboard_{os.path.basename(self.simulation_dir)}.mp4')
        
        # 첫 번째 프레임 로드하여 비디오 크기 설정
        first_frame = cv2.imread(dashboard_frames[0])
        height, width, layers = first_frame.shape
        
        # 비디오 코덱 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 코덱
        video = cv2.VideoWriter(video_path, fourcc, frame_rate, (width, height))
        
        print("Creating dashboard video...")
        for frame_path in tqdm(dashboard_frames):
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        # 비디오 저장
        video.release()
        
        print(f"Dashboard video saved to {video_path}")
        
        # 임시 프레임 삭제
        if delete_frames:
            for frame_path in dashboard_frames:
                os.remove(frame_path)
            os.rmdir(temp_dir)
            print("Temporary frames deleted")
        
        return video_path


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
    
    # 비디오 생성기 초기화
    video_creator = VideoCreator(latest_dir)
    
    # 1. 기본 맵 비디오 생성
    # video_path = video_creator.create_video(frame_rate=10)
    
    # 2. 감염 통계 비디오 생성
    # stats_video_path = video_creator.create_stats_video(frame_rate=10)
    
    # 3. 맵과 통계를 나란히 보여주는 비디오 생성
    # combined_video_path = video_creator.create_side_by_side_video(frame_rate=10)
    
    # 4. 대시보드 형태의 비디오 생성
    dashboard_video_path = video_creator.create_dashboard_video(frame_rate=10)
    
    print("All videos have been created successfully!")
