import pandas as pd

dataset_dir = "../../data/"

df = None
for day in range(1, 7 + 1):
    invocations_per_function = dataset_dir + \
        'invocations_per_function_md.anon.d0' + str(day) + '.csv'
    
    df_day = pd.read_csv(invocations_per_function)
    if df is None:
        df = df_day
    else:
        df = pd.concat([df, df_day], ignore_index=True)

num_cols = [str(i) for i in range(1, 1441)]
df['total'] = df[num_cols].sum(axis=1)

# Step 3: group by HashFunction and sum all totals
hash_sums = df.groupby('HashFunction')['total'].sum().sort_values(ascending=False)

# Step 4: save the results to a new CSV
# hash_sums.to_csv("hashfunction_invocation_sums.csv", header=['TotalSum'])

filtered = hash_sums[(hash_sums >= 80640) & (hash_sums <= 81000)]
print(filtered.head(5))

# print(hash_sums.head(5))  # optional: preview top few