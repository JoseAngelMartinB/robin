import os

from robin.kernel.entities import Kernel

os.makedirs('data/kernel_output', exist_ok=True)

path_config_supply = 'configs/supply_data.yml'
path_config_demand = 'configs/demand_data.yml'
seed = 0

kernel = Kernel(path_config_supply, path_config_demand, seed)
services = kernel.simulate('data/kernel_output/output.csv', departure_time_hard_restriction=True)

for service in services:
    print(service)
