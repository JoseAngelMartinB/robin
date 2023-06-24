from src.robin.scraping.entities import DataLoader, SupplySaver

import_path = 'data/renfe'
trips_path = f'{import_path}/stop_times/stopTimes_MADRI_BARCE_2023-06-23_2023-06-24.csv'

data_loader = DataLoader(trips_path)
data_loader.show_metadata()

data_loader.build_supply_entities()
print(data_loader.services[0])

services = data_loader.services
SupplySaver(services).to_yaml(filename='supply_data_23Jun.yml', save_path='configs/')
