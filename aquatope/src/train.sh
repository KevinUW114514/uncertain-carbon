mkdir model_artifacts

python train_lstm_encoder_decoder.py \
    --n_input_steps 48 \
    --n_output_steps 12 \
    --num_days 7 \
    --num_epochs 128 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --variational_dropout_p 0.25 \
    --trace_id "0533d1cd0ba44d166a0567b8595b497a3eb917fb06e74cea43c5292d222c8dc9" \
    --dataset_dir "../../data/"

python train_prediction_network.py \
    --n_input_steps 48 \
    --n_output_steps 1 \
    --num_days 7 \
    --num_epochs 128 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --dropout_p 0.25 \
    --trace_id "0533d1cd0ba44d166a0567b8595b497a3eb917fb06e74cea43c5292d222c8dc9" \
    --dataset_dir "../../data/"