import datetime
import time


def parse_time(config: dict) -> datetime.datetime:
    """
    time: dic -> datetime.datetime
    :param config: dict
    :return: datetime.datetime

    Examples:
    >>> dic = {"year": 2022, "month": 2, "day": 21, 'hour': 10, 'minute': 0, 'second': 0}
    >>> parse_time(dic)
    2022-02-21 10:00:00

    >>> dic = {"year": 2022, "month": 2, "day": 21}
    >>> parse_time(dic)
    2022-02-21 00:00:00
    """
    year = config.get('year')
    month = config.get('month')
    day = config.get('day')
    hour = config.get('hour', 0)
    minute = config.get('minute', 0)
    second = config.get('second', 0)
    t = datetime.datetime(year, month, day, hour, minute, second)
    return t


def datetime_to_mktime(t: datetime.datetime):
    return int(time.mktime(t.timetuple()))
