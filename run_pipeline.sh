#!/bin/bash
set -e
cd /home/user/pro2

echo "=== STEP 1: nautilus pass 1 ==="
python3 nautilus_4agent.py --checkpoint hmoe_self_trained_v5.pt --cycles 8 --step-scale 0.4 --save pipeline_runs/from_mid/pass1.pt 2>&1
echo "=== PASS1 DONE ==="

echo "=== STEP 2: nautilus pass 2 ==="
python3 nautilus_4agent.py --checkpoint pipeline_runs/from_mid/pass1.pt --cycles 8 --step-scale 0.4 --save pipeline_runs/from_mid/pass2.pt 2>&1
echo "=== PASS2 DONE ==="

echo "=== STEP 3: turbine ==="
python3 figure8_turbine.py --checkpoint pipeline_runs/from_mid/pass2.pt --cycles 8 --steps_per_expert 8 --lci-loss 0.1 --save pipeline_runs/from_mid/turbine.pt 2>&1
echo "=== TURBINE DONE ==="

echo "=== STEP 4: bench ==="
python3 bench_all.py --checkpoint pipeline_runs/from_mid/turbine.pt --variants 8,5,6 2>&1
echo "=== BENCH DONE ==="
