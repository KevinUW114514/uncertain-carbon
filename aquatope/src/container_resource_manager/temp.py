# import json

# resource_config = [{'cpu_m': 355, 'memory_mi': 70}, {'cpu_m': 5410, 'memory_mi': 512}]
# with open("best_resource_config_default_1.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 62}, {'cpu_m': 4708, 'memory_mi': 418}]
# with open("best_resource_config_default_2.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 70}, {'cpu_m': 4935, 'memory_mi': 509}]
# with open("best_resource_config_default_3.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 70}, {'cpu_m': 5481, 'memory_mi': 486}]
# with open("best_resource_config_energy_1.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 50}, {'cpu_m': 6000, 'memory_mi': 400}]
# with open("best_resource_config_energy_2.json", "w") as f:
#     json.dump(resource_config, f, indent=2)    
    
# resource_config = [{'cpu_m': 300, 'memory_mi': 64}, {'cpu_m': 6000, 'memory_mi': 400}]
# with open("best_resource_config_energy_3.json", "w") as f:
#     json.dump(resource_config, f, indent=2)
"PYTHONPATH=/home/cc/uncertain-carbon/aquatope:$PYTHONPATH"
import locust
import pandas as pd
from fissionlib.cli import LOCUST_CSV, start_locust, stop_locust
import time

# start_locust()
# time.sleep(10)  # wait for locust to start
# stop_locust()

start = 1767744712
locust_csv_df = pd.read_csv(f"{LOCUST_CSV}_stats_history.csv")
start_requests = locust_csv_df.loc[
    locust_csv_df["Timestamp"] == start,
    "Total Request Count"
].iloc[0]
print(type(start_requests))   # pandas Series
print(start_requests)