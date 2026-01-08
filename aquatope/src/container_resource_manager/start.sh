for i in {1..5}; do
    ts=$(date +"%Y%m%d_%H%M%S")
    log="result_${ts}.log"

    pkill -9 -f locust

    : > "$log"
    echo "Run $i" | tee -a "$log"
    # echo "resume"
    # echo "========================================" | tee -a "$log"

    echo "========================================" | tee -a "$log" || true
    echo "Default" | tee -a "$log"
    IS_ENERGY=0 python -u manager.py --n_batch 10  2>&1 | tee -a "$log"

    echo "========================================" | tee -a "$log" || true
    echo "Energy" | tee -a "$log"
    IS_ENERGY=1 python -u manager.py --n_batch 10  2>&1 | tee -a "$log"
    echo "========================================" | tee -a "$log"
done