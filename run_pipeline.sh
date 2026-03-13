#!/bin/bash
# Auto-chain: wait for stage2_default.pth → eval → ablate → visualize → update_report

PYTHON=/home/koe/miniconda/bin/python
BASE=/data/koe/ECE285-Final

echo "=== Pipeline runner started at $(date) ==="

# Wait for Stage 2 default to complete (history file written last)
echo "Waiting for Stage 2 default to complete..."
while [ ! -f "$BASE/results/stage2_default_history.json" ]; do
    sleep 30
done
echo "Stage 2 default done at $(date)"

# Step 1: Evaluate default model
echo ""
echo "=== Evaluation ==="
cd "$BASE" && $PYTHON src/eval.py --ckpt checkpoints/stage2_default.pth --tag default
echo "Eval done at $(date)"

# Step 2: Run ablations
echo ""
echo "=== Ablations ==="
cd "$BASE" && $PYTHON src/ablate.py
echo "Ablations done at $(date)"

# Step 3: Generate visualizations
echo ""
echo "=== Visualizations ==="
cd "$BASE" && $PYTHON src/visualize.py
echo "Visualize done at $(date)"

# Step 4: Update report macros
echo ""
echo "=== Updating report ==="
cd "$BASE" && $PYTHON update_report.py
echo "Report updated at $(date)"

# Step 5: Compile report PDF and slides
echo ""
echo "=== Compiling PDFs ==="
cd "$BASE" && pdflatex -interaction=nonstopmode report.tex > /tmp/pdflatex.log 2>&1
cd "$BASE" && pdflatex -interaction=nonstopmode report.tex >> /tmp/pdflatex.log 2>&1
cd "$BASE" && pdflatex -interaction=nonstopmode slides.tex >> /tmp/pdflatex.log 2>&1
echo "PDFs compiled at $(date)"
grep "Output written" /tmp/pdflatex.log | tail -2

echo ""
echo "=== All done at $(date) ==="
