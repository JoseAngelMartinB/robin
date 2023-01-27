class InvalidDistributionException(Exception):
    """Raised when the given distribution is not contained in SciPy."""
    pass


class Station:
    """
    Dummy class for a station.
    
    NOTE: This class is not yet implemented. It is just a placeholder.
    """
    pass


class Market:
    """
    A market is composed by the departure and arrival stations.

    Attributes:
        id (int): The market id.
        departure_station (Station): The departure station.
        arrival_station (Station): The arrival station.
    """

    def __init__(self, id: int, departure_station: Station, arrival_station: Station) -> None:
        self.id = id
        self.departure_station = departure_station
        self.arrival_station = arrival_station


class UserPattern:
    """A user pattern is a set of random variables and penalties that define it (e.g. business or student)."""
    pass


class DemandPattern:
    """A demand pattern is determined by the potential demand and the distribution of user patterns."""
    pass


class Day:
    """A day is described as its actual date and demand pattern."""
    pass


class Passenger:
    """A passenger is defined by his/her user pattern, the origin-destination pair, the desired day and time of arrival and the day of purchase."""
    pass
