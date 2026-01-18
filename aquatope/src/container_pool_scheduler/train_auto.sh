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

python data_processing.py --column 6 --path /home/kevin/research/uncertain-carbon/data/requests_minute -t -v -c
python data_processing.py --column 28 --path /home/kevin/research/uncertain-carbon/data/requests_minute -t -v -c

python train_lstm_encoder_decoder.py \
    --n_input_steps 24 \
    --n_output_steps 12 \
    --num_days 7 \
    --num_epochs 200 \
    --batch_size 32 \
    --learning_rate 3e-4 \
    --variational_dropout_p 0 \
    --dataset_dir "./"

python train_prediction_network.py \
    --n_input_steps 24 \
    --n_output_steps 1 \
    --num_days 7 \
    --num_epochs 200 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --dropout_p 0 \
    --dataset_dir "./"

python full_inference.py \
    --n_input_steps 24 \
    --n_output_steps 1 \
    --dataset_dir "./"


# max error rate: 9.486397743225098
# ================================================================================
# Overall mean error rate: 52.99%
# ================================================================================
# 25th percentile error rate: 7.84%
# 50th percentile error rate: 18.23%
# 75th percentile error rate: 34.24%
# 90th percentile error rate: 140.83%
# 95th percentile error rate: 277.19%
# 99th percentile error rate: 481.93%
# 99.9th percentile error rate: 940.32%
# 99.99th percentile error rate: 947.81%
# [inference] mean: 0.08152421563863754, var: 0.077083058655262, smape_rate: 30.10874366760254
# predicted.shape: torch.Size([730]), target.shape: torch.Size([730]), error_rates.shape: torch.Size([730])
# x.shape: torch.Size([730, 60, 1]), y.shape: torch.Size([730, 1, 3])
# predicted.shape: torch.Size([730]), target.shape: torch.Size([730]), error_rates.shape: torch.Size([730])
# max target: 244.0, min target: 10.0
# max target: 244.0, min target: 10.0
# max error rate: 2.0850937366485596
# ================================================================================
# Overall mean error rate: 20.01%
# ================================================================================
# 25th percentile error rate: 6.84%
# 50th percentile error rate: 15.85%
# 75th percentile error rate: 26.36%
# 90th percentile error rate: 39.84%
# 95th percentile error rate: 52.61%
# 99th percentile error rate: 104.53%
# 99.9th percentile error rate: 188.95%
# 99.99th percentile error rate: 206.55%
# [inference_conformal_regime_conditioned] mean_abs_err: 21.211689, mean_model_var: 0.000000, smape_rate: 18.941492080688477
# Traceback (most recent call last):
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 72, in <module>
#     main()
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 65, in main
#     mean, var = utils.inference_conformal(datasets=datasets, model=predict, mc_dropout=False)
# ValueError: too many values to unpack (expected 2)




# 730 samples of 60 input steps and 1 output steps in inference
# /home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/utils.py:1650: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   mean = torch.mean(torch.tensor(res)).to(device)
# /home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/utils.py:1651: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone() or sourceTensor.detach().clone().requires_grad_(True), rather than torch.tensor(sourceTensor).
#   var = torch.var(torch.tensor(res))
# max error rate: 3.5747008323669434
# ================================================================================
# Overall mean error rate: 26.85%
# ================================================================================
# 25th percentile error rate: 9.07%
# 50th percentile error rate: 17.42%
# 75th percentile error rate: 29.24%
# 90th percentile error rate: 46.61%
# 95th percentile error rate: 99.73%
# 99th percentile error rate: 174.21%
# 99.9th percentile error rate: 351.52%
# 99.99th percentile error rate: 356.88%
# [inference] mean: -0.2029625028371811, var: 0.18442347645759583, smape_rate: 23.298110961914062
# predicted.shape: torch.Size([730]), target.shape: torch.Size([730]), error_rates.shape: torch.Size([730])
# x.shape: torch.Size([730, 60, 1]), y.shape: torch.Size([730, 1, 3])
# predicted.shape: torch.Size([730]), target.shape: torch.Size([730]), error_rates.shape: torch.Size([730])
# max target: 244.0, min target: 10.0
# max target: 244.0, min target: 10.0
# max error rate: 2.299483299255371
# ================================================================================
# Overall mean error rate: 19.83%
# ================================================================================
# 25th percentile error rate: 7.01%
# 50th percentile error rate: 15.07%
# 75th percentile error rate: 26.57%
# 90th percentile error rate: 39.59%
# 95th percentile error rate: 50.98%
# 99th percentile error rate: 79.83%
# 99.9th percentile error rate: 212.77%
# 99.99th percentile error rate: 228.23%
# [inference_conformal_regime_conditioned] mean_abs_err: 21.170633, mean_model_var: 0.000000, smape_rate: 19.03520393371582
# Traceback (most recent call last):
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 72, in <module>
#     main()
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 65, in main
#     mean, var = utils.inference_conformal(datasets=datasets, model=predict, mc_dropout=False)


# max error rate: 10.101007461547852
# ================================================================================
# Overall mean error rate: 65.96%
# ================================================================================
# 25th percentile error rate: 12.45%
# 50th percentile error rate: 24.21%
# 75th percentile error rate: 40.99%
# 90th percentile error rate: 191.92%
# 95th percentile error rate: 352.06%
# 99th percentile error rate: 682.41%
# 99.9th percentile error rate: 999.62%
# 99.99th percentile error rate: 1009.13%
# [inference] mean: -0.05457206815481186, var: 0.10835174471139908, smape_rate: 37.070560455322266
# predicted.shape: torch.Size([1072]), target.shape: torch.Size([1072]), error_rates.shape: torch.Size([1072])
# x.shape: torch.Size([1072, 15, 1]), y.shape: torch.Size([1072, 1, 3])
# predicted.shape: torch.Size([1072]), target.shape: torch.Size([1072]), error_rates.shape: torch.Size([1072])
# max target: 259.0, min target: 10.0
# max target: 259.0, min target: 10.0
# max error rate: 2.070549249649048
# ================================================================================
# Overall mean error rate: 20.33%
# ================================================================================
# 25th percentile error rate: 6.78%
# 50th percentile error rate: 15.19%
# 75th percentile error rate: 27.47%
# 90th percentile error rate: 42.49%
# 95th percentile error rate: 53.79%
# 99th percentile error rate: 99.84%
# 99.9th percentile error rate: 196.26%
# 99.99th percentile error rate: 206.05%
# [inference_conformal_regime_conditioned] mean_abs_err: 21.351513, mean_model_var: 0.000000, smape_rate: 19.517860412597656
# Traceback (most recent call last):
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 72, in <module>
#     main()
#   File "/home/kevin/research/uncertain-carbon/aquatope/src/container_pool_scheduler/full_inference.py", line 65, in main
#     mean, var = utils.inference_conformal(datasets=datasets, model=predict, mc_dropout=False)
# ValueError: too many values to unpack (expected 2)