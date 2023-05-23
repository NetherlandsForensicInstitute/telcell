from __future__ import annotations
from abc import abstractmethod


class Item:
    """
    An instance of a database item.
    """

    def __init__(self, **extra):
        self.extra = extra

    def as_dict(self) -> dict:
        return dict(self.extra)

    def with_value(self, **values) -> Item:
        d = self.as_dict()
        for key, value in values.items():
            if callable(value):
                value = value(self)
            d[key] = value
        return self.create_item(**d)

    @abstractmethod
    def create_item(self, **values) -> Item:
        raise NotImplementedError

    def __getattr__(self, item):
        if item in self.extra:
            return self.extra[item]
        else:
            raise AttributeError(item)

    def __eq__(self, other):
        return type(self) == type(other) and self.extra == other.extra

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra})"
