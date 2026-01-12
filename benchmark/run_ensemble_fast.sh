#!/bin/bash
# Fast ensemble test - only the best strategy (Simple Average)

echo "================================================"
echo "Fast Ensemble Test"
echo "================================================"
echo ""
echo "Running only Simple Average Ensemble"
echo "This is usually the best performing ensemble strategy"
echo ""

echo "Combining: Random Forest + LightGBM + XGBoost + Ridge"
echo "Strategy: Simple Average (equal weights)"
echo ""

uv run python ml_ensemble.py --strategy mean \
    --models random_forest lightgbm xgboost ridge

echo ""
echo "================================================"
echo "✅ Ensemble Completed!"
echo "================================================"
echo ""
echo "Results saved to:"
echo "  ../results/ensemble_mean_random_forest_lightgbm_xgboost_ridge/all_results.csv"
echo ""
echo "To view:"
echo "  cat ../results/ensemble_mean_random_forest_lightgbm_xgboost_ridge/all_results.csv"
