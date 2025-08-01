# COVID-19 시뮬레이션 병렬화 (MPI) 가이드

## 병렬화 개요

이 프로젝트는 COVID-19 전염병 확산 시뮬레이션을 MPI(Message Passing Interface)를 활용하여 병렬화한 버전입니다. 주요 병렬화 전략과 구현 내용은 다음과 같습니다.

### 1. 영역 분할 (Domain Decomposition)

맵을 여러 영역으로 나누고 각 MPI 프로세스가 특정 영역을 담당하는 방식으로 구현했습니다:
- 맵을 행(row) 기준으로 분할하여 각 프로세스가 담당
- 각 프로세스는 자신의 영역에 있는 사람들만 처리
- 경계를 넘어가는 사람들은 프로세스 간 데이터 교환을 통해 처리

### 2. 병렬화된 주요 기능

- **사람 이동 시뮬레이션**: 각 프로세스가 자신의 영역 내 사람들의 이동을 독립적으로 계산
- **감염 시뮬레이션**: 각 프로세스가 자신의 영역 내 감염 확산을 독립적으로 계산
- **경계 데이터 교환**: 영역을 넘어가는 사람들에 대한 정보를 프로세스 간 교환
- **통계 수집**: 각 프로세스의 로컬 통계를 모아 전체 통계를 계산

### 3. 병렬화된 핵심 함수

- `split_people_by_regions()`: 초기 사람들을 영역별로 분배
- `exchange_boundary_data()`: 경계를 넘어가는 사람들에 대한 정보 교환
- `gather_all_people()`: 모든 프로세스의 사람 데이터를 수집
- `run_simulation_parallel()`: 병렬화된 시뮬레이션 실행

## 실행 방법

MPI로 시뮬레이션을 실행하려면 다음 명령어를 사용하세요:

```bash
mpiexec -n 4 python main.py
```

여기서 `-n 4`는 4개의 프로세스를 사용한다는 의미입니다. 이 값은 사용 가능한 코어 수에 맞게 조정할 수 있습니다.

## 성능 최적화 팁

1. **프로세스 수 조정**: 맵 크기와 사람 수에 따라 최적의 프로세스 수가 달라질 수 있습니다.
2. **영역 크기 균형**: 인구 밀도가 높은 지역과 낮은 지역의 균형을 맞추기 위해 동적 로드 밸런싱을 고려할 수 있습니다.
3. **통신 최적화**: 경계 교환 시 필요한 데이터만 교환하도록 최적화가 가능합니다.
4. **병렬 I/O**: 대규모 시뮬레이션에서는 MPI I/O를 사용하여 파일 저장을 병렬화할 수 있습니다.
