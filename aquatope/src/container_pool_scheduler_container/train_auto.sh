# python train_lstm_encoder_decoder.py \
#     --n_input_steps 60 \
#     --n_output_steps 5 \
#     --num_epochs 350 \
#     --batch_size 128 \
#     --learning_rate 3e-4 \
#     --variational_dropout_p 0.15 \
#     --dataset_dir "./"


# python train_lstm_encoder_decoder.py \
#     --n_input_steps 48 \
#     --n_output_steps 12 \
#     --num_days 7 \
#     --num_epochs 128 \
#     --batch_size 128 \
#     --learning_rate 1e-4 \
#     --variational_dropout_p 0.25 \
#     --dataset_dir "./"

# python train_prediction_network.py \
#     --n_input_steps 48 \
#     --n_output_steps 1 \
#     --num_days 7 \
#     --num_epochs 128 \
#     --batch_size 128 \
#     --learning_rate 1e-3 \
#     --dropout_p 0.25 \
#     --dataset_dir "./"

# python full_inference.py \
#     --n_input_steps 48 \
#     --n_output_steps 1 \
#     --dataset_dir "./"

python data_processing.py --column 6 --path /home/kevin/research/uncertain-carbon/data/requests_minute -t -v

python train_lstm_encoder_decoder.py \
    --n_input_steps 60 \
    --n_output_steps 5 \
    --num_days 7 \
    --num_epochs 50 \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --variational_dropout_p 0.2 \
    --dataset_dir "./"

python train_prediction_network.py \
    --n_input_steps 60 \
    --n_output_steps 1 \
    --num_days 7 \
    --num_epochs 50 \
    --batch_size 128 \
    --learning_rate 1e-3 \
    --dropout_p 0.2 \
    --dataset_dir "./"

python full_inference.py \
    --n_input_steps 60 \
    --n_output_steps 1 \
    --dataset_dir "./"