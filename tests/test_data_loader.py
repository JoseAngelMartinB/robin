from src.robin.scraping.entities import DataLoader, IMPORT_PATH

trips_path = f'{IMPORT_PATH}/trips/trips_MADRI_BARCE_2023-06-01_2023-06-02.csv'

data_loader = DataLoader(trips_path)
data_loader.show_metadata()

data_loader.build_supply_entities()
print(list(data_loader.services.values())[1])

data_loader.save_yaml(save_path='configs/test_case/supply_data.yml')
