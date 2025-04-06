from robin.labs.entities import RobinLab

config = {"supply": {"start": 0.0, "stop": 10.0, "step": 1.0},
          "demand": ()}

robin_lab = RobinLab(path_config_supply="configs/test_case/supply_data.yml",
                     path_config_demand="configs/test_case/demand_data.yml",
                     tmp_path="data/labs/tmp")

robin_lab.set_lab_config(config=config)
robin_lab.simulate()
