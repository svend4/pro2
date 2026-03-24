#!/bin/bash
# A/B test: same 3 checkpoints but with meta_q6 improvements
# --bent-seeds: bent-functions from Q6 as seed archetypes (nl=28, WHT uniform)
# --temp-decay 0.85: Metropolis temperature annealing (reduces oscillations ±0.05)
set -e
export OMP_NUM_THREADS=1
cd /home/user/pro2

echo "=== META_Q6 A/B TEST PIPELINE ===" | tee /tmp/ab_test_pipeline.log
date | tee -a /tmp/ab_test_pipeline.log
echo "Flags: --bent-seeds --temp-decay 0.85" | tee -a /tmp/ab_test_pipeline.log

# ── WEAK checkpoint (hmoe_self_trained_v4.pt) ─────────────────────────────────
echo "" && echo "=== WEAK_v2: nautilus pass 1 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_self_trained_v4.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_weak_v2/pass1.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== WEAK_v2: nautilus pass 2 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_weak_v2/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_weak_v2/pass2.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== WEAK_v2: figure8 turbine (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_weak_v2/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --temp-decay 0.85 \
  --save pipeline_runs/from_weak_v2/turbine.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== WEAK_v2: bench_all ===" | tee -a /tmp/ab_test_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_weak_v2/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/ab_test_pipeline.log

# ── MID checkpoint (hmoe_self_trained_v5.pt) ─────────────────────────────────
echo "" && echo "=== MID_v2: nautilus pass 1 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_self_trained_v5.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_mid_v2/pass1.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== MID_v2: nautilus pass 2 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_mid_v2/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_mid_v2/pass2.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== MID_v2: figure8 turbine (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_mid_v2/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --temp-decay 0.85 \
  --save pipeline_runs/from_mid_v2/turbine.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== MID_v2: bench_all ===" | tee -a /tmp/ab_test_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_mid_v2/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/ab_test_pipeline.log

# ── STRONG checkpoint (hmoe_v2_self.pt) ──────────────────────────────────────
echo "" && echo "=== STRONG_v2: nautilus pass 1 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint hmoe_v2_self.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_strong_v2/pass1.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== STRONG_v2: nautilus pass 2 (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 nautilus_4agent.py \
  --checkpoint pipeline_runs/from_strong_v2/pass1.pt \
  --cycles 8 --step-scale 0.4 \
  --bent-seeds --temp-decay 0.85 \
  --save pipeline_runs/from_strong_v2/pass2.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== STRONG_v2: figure8 turbine (meta_q6) ===" | tee -a /tmp/ab_test_pipeline.log
python3 figure8_turbine.py \
  --checkpoint pipeline_runs/from_strong_v2/pass2.pt \
  --cycles 8 --steps_per_expert 8 \
  --temp-decay 0.85 \
  --save pipeline_runs/from_strong_v2/turbine.pt 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "=== STRONG_v2: bench_all ===" | tee -a /tmp/ab_test_pipeline.log
python3 bench_all.py \
  --checkpoint pipeline_runs/from_strong_v2/turbine.pt \
  --variants 8,5,6 2>&1 | tee -a /tmp/ab_test_pipeline.log

echo "" && echo "=== ALL A/B TESTS COMPLETE ===" | tee -a /tmp/ab_test_pipeline.log
date | tee -a /tmp/ab_test_pipeline.log
