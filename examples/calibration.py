from robin.calibration.entities import Calibration

calibration = Calibration(
    path_config_supply='configs/calibration/supply_data.yml',
    path_config_demand='configs/calibration/demand_data.yml',
    target_output_path='configs/calibration/target.csv',
    seed=300
)
calibration.create_study(
    study_name='calibration_test',
    storage='sqlite:///calibration_test.db',
    n_trials=100,
    show_progress_bar=True
)
