price_models=(
    bo_model_price_20260105_103344.pt
    bo_model_price_20260105_084828.pt
    bo_model_price_20260105_070411.pt
    bo_model_price_20260105_051239.pt
)

energy_models=(
    bo_model_energy_20260105_112623.pt
    bo_model_energy_20260105_094046.pt
    bo_model_energy_20260105_075601.pt
    bo_model_energy_20260105_061317.pt
)

for i in "${!price_models[@]}"; do
    ts=$(date +"%Y%m%d_%H%M%S")
    log="resume_result_${ts}.log"

    pkill -9 -f locust

    : > "$log"
    echo "Run $i" | tee -a "$log"
    echo "resume"
    echo "========================================" | tee -a "$log"

    echo "========================================" | tee -a "$log"
    echo "Default" | tee -a "$log"
    IS_ENERGY=0 python -u manager.py \
        --n_batch 10 \
        --model_path "${price_models[$i]}" \
        2>&1 | tee -a "$log"

    echo "========================================" | tee -a "$log"
    echo "Energy" | tee -a "$log"
    IS_ENERGY=1 python -u manager.py \
        --n_batch 10 \
        --model_path "${energy_models[$i]}" \
        2>&1 | tee -a "$log"

    echo "========================================" | tee -a "$log"
done
