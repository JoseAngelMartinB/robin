class InvalidDistributionException(Exception):
    """Raised when the given distribution is not contained in SciPy."""
    pass


class Market:
    """A market is composed by the origin and destination stations."""
    pass


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
