from robin.scraping.entities import DataLoader, SupplySaver

import_path = 'data/renfe'
trips_path = f'{import_path}/stop_times/stopTimes_MADRI_BARCE_2023-09-06_2023-09-07.csv'

data_loader = DataLoader(trips_path)
data_loader.show_metadata()

data_loader.build_supply_entities()
print(data_loader.services[0])

services = data_loader.services
SupplySaver(services).to_yaml(filename='supply_data_increased_prices.yml', save_path='configs/ewgt23/')
