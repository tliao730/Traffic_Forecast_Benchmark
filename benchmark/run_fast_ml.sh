#!/bin/bash
# Run only fast ML methods (completes in ~6-7 hours)

echo "================================================"
echo "Fast ML Methods Evaluation"
echo "================================================"
echo ""
echo "This script runs only the FAST ML methods."
echo "Estimated total time: 6-7 hours"
echo ""
echo "Skipping: LightGBM (~118 hours), XGBoost (~166 hours)"
echo ""

# 1. Seasonal Naive (~0.1 hours)
echo "================================================"
echo "1/4: Seasonal Naive (baseline) - ~5 minutes"
echo "================================================"
uv run python ml_methods.py --model seasonal_naive
echo ""

# 2. Linear Regression (~0.3 hours)
echo "================================================"
echo "2/4: Linear Regression - ~20 minutes"
echo "================================================"
uv run python ml_methods.py --model linear_regression
echo ""

# 3. Ridge (~0.2 hours)
echo "================================================"
echo "3/4: Ridge Regression - ~15 minutes"
echo "================================================"
uv run python ml_methods.py --model ridge
echo ""

# 4. Random Forest (~6 hours)
echo "================================================"
echo "4/4: Random Forest - ~6 hours"
echo "================================================"
echo "This is the slowest, but still reasonable."
uv run python ml_methods.py --model random_forest
echo ""

echo "================================================"
echo "✅ Fast ML Methods Completed!"
echo "================================================"
echo ""
echo "Results saved in:"
echo "  ../results/seasonal_naive/all_results.csv"
echo "  ../results/linear_regression/all_results.csv"
echo "  ../results/ridge/all_results.csv"
echo "  ../results/random_forest/all_results.csv"
echo ""
echo "To compare results:"
echo "  uv run python compare_ml_results.py"
echo ""
echo "To create an ensemble of these fast models:"
echo "  uv run python ml_ensemble.py --strategy mean \\"
echo "    --models seasonal_naive ridge linear_regression random_forest"
