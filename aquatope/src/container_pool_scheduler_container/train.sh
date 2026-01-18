python data_processing.py --column 6 --path /home/kevin/research/uncertain-carbon/data/requests_minute -t -v


python train_lstm_encoder_decoder.py \
    --n_input_steps 60 \
    --n_output_steps 5 \
    --num_epochs 350 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --variational_dropout_p 0.15 \
    --dataset_dir "./"

python train_prediction_network.py \
    --n_input_steps 60 \
    --n_output_steps 1 \
    --num_epochs 350 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --dropout_p 0.15 \
    --dataset_dir "./"

# 100 rps: 
python train_lstm_encoder_decoder.py \
    --n_input_steps 24 \
    --n_output_steps 2 \
    --num_days 7 \
    --num_epochs 350 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --variational_dropout_p 0.15 \
    --trace_id "e896d7ac37090135a8a1c812e6d3a9a64d15e7806332d189935e9e593fab6322" \
    --dataset_dir "./"

python train_prediction_network.py \
    --n_input_steps 24 \
    --n_output_steps 1 \
    --num_days 7 \
    --num_epochs 300 \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --dropout_p 0.15 \
    --trace_id "0533d1cd0ba44d166a0567b8595b497a3eb917fb06e74cea43c5292d222c8dc9" \
    --dataset_dir "./"

python full_inference.py \
    --n_input_steps 60 \
    --n_output_steps 1 \
    --dataset_dir "./"

python ts_inference.py --n_input_steps 60 --n_output_steps 60

python new_predictor.py \
  --data_path train_samples.csv \
  --input_n 60 \
  --output_n 60 \
  --hidden_size 128 \
  --num_layers 2 \
  --dropout_in 0.1 \
  --dropout_hidden 0.2 \
  --batch_size 128 \
  --epochs 20 \
  --mc_samples 50

python infer.py \
  --data_path train_samples.csv \
  --ckpt_path model.pt \
  --mc_samples 50 \
  --windows 1 \
  --include_z \
  --out_csv inference_with_rps.csv


