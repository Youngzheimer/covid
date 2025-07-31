#!/bin/bash

# 성능 디버깅을 위한 실행 스크립트
# MPI 환경 변수 설정
export OMPI_MCA_btl_base_verbose=100  # MPI 통신 상세 로그
export OMPI_MCA_mpi_yield_when_idle=1  # CPU 사용 최적화

# 프로세스 수를 인자로 받거나 기본값 4 사용
NUM_PROCESSES=${1:-4}

echo "Starting COVID-19 simulation with $NUM_PROCESSES MPI processes"

# MPI를 사용하여 시뮬레이션 실행
# -v 옵션: 상세 출력
mpiexec -v -n $NUM_PROCESSES python main.py 2>&1 | tee mpi_debug_log.txt

echo "Simulation complete. Debug log saved to mpi_debug_log.txt"

# 사용 예:
# ./debug_parallel_simulation.sh 8     # 8개의 프로세스로 실행
# ./debug_parallel_simulation.sh       # 기본값(4개 프로세스)로 실행
