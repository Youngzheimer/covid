#!/bin/bash

# 프로세스 수를 인자로 받거나 기본값 4 사용
NUM_PROCESSES=${1:-4}

# MPI를 사용하여 시뮬레이션 실행
mpiexec -n $NUM_PROCESSES python main.py

# 사용 예:
# ./run_parallel_simulation.sh 8     # 8개의 프로세스로 실행
# ./run_parallel_simulation.sh       # 기본값(4개 프로세스)로 실행
