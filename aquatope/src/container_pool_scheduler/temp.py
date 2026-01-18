import pickle

rps = []
with open("r_samp_history.pkl", "rb") as f:
    while True:
        try:
            rps.append(pickle.load(f))
        except EOFError:
            break
rps = rps[:336]
print(len(rps))

ci = []
with open("GB_direct_rolling_t1_eval_residual_samples.pkl", "rb") as f:
    ci = pickle.load(f)
print(len(ci))


