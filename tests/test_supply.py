from src.robin.supply.entities import Supply

supply = Supply.from_yaml('configs/supply_data_error.yml')
service = supply.filter_service_by_id('1')
print(service)

# Case 1
"""
print(service.buy_ticket('MAD', 'ZAZ', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('MAD', 'ZAZ', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('MAD', 'BCN', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('ZAZ', 'BCN', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('ZAZ', 'BCN', service.seat_types['First class - With luggage'], 0))
"""

# Case 2 (Error)
print(service.buy_ticket('MAD', 'ZAZ', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('MAD', 'ZAZ', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('ZAZ', 'BCN', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('ZAZ', 'BCN', service.seat_types['First class - With luggage'], 0))
print(service.buy_ticket('MAD', 'BCN', service.seat_types['First class - With luggage'], 0))
