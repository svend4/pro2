#!/bin/bash
# Run all 3 baseline pipeline passes sequentially with OMP_NUM_THREADS=1
# This prevents thread contention that caused 87-minute hangs
set -e
export OMP_NUM_THREADS=1
cd /home/user/pro2

echo "=== BASELINE SEQUENTIAL PIPELINE ===" | tee /tmp/baseline_pipeline.log
date | tee -a /tmp/baseline_pipeline.log

# ── WEAK checkpoint (hmoe_self_trained_v4.pt, LCI≈2.33) ─────────────────────
echo "" && echo "=== WEAK: nautilus pass 1 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_self_trained_v4.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_weak/pass1.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== WEAK: nautilus pass 2 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_weak/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_weak/pass2.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== WEAK: figure8 turbine ===" | tee -a /tmp/baseline_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_weak/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --save pipeline_runs/from_weak/turbine.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== WEAK: bench_all ===" | tee -a /tmp/baseline_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_weak/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/baseline_pipeline.log

# ── MID checkpoint (hmoe_self_trained_v5.pt, LCI≈2.978) ─────────────────────
echo "" && echo "=== MID: nautilus pass 1 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_self_trained_v5.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_mid/pass1.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== MID: nautilus pass 2 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_mid/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_mid/pass2.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== MID: figure8 turbine ===" | tee -a /tmp/baseline_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_mid/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --save pipeline_runs/from_mid/turbine.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== MID: bench_all ===" | tee -a /tmp/baseline_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_mid/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/baseline_pipeline.log

# ── STRONG checkpoint (hmoe_v2_self.pt, LCI=3.135) ──────────────────────────
echo "" && echo "=== STRONG: nautilus pass 1 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_v2_self.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_strong/pass1.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== STRONG: nautilus pass 2 ===" | tee -a /tmp/baseline_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_strong/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --save pipeline_runs/from_strong/pass2.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== STRONG: figure8 turbine ===" | tee -a /tmp/baseline_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_strong/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --save pipeline_runs/from_strong/turbine.pt 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "=== STRONG: bench_all ===" | tee -a /tmp/baseline_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_strong/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/baseline_pipeline.log

echo "" && echo "=== ALL BASELINES COMPLETE ===" | tee -a /tmp/baseline_pipeline.log
date | tee -a /tmp/baseline_pipeline.log
