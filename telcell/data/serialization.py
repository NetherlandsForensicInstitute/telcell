import csv
import datetime
import sys
from abc import abstractmethod, ABC
from typing import Callable, Any, Optional, Iterable, Tuple, Iterator

from .cell_identity import CellIdentity
from .coord import WgsPoint, Point
from .geography import Angle
from .measurement import CellMeasurementSet, CellMeasurement
from .properties import Properties, LocationInfo


class Dictionizer(ABC):
    @abstractmethod
    def is_dictable(self, key: str, value: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def to_dict(self, key: str, value: Any) -> Tuple[str, Any]:
        raise NotImplementedError


class Dedictionizer(ABC):
    @abstractmethod
    def is_dedictable(self, key: str, value: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def from_dict(self, key: str, value: Any) -> Tuple[str, Any]:
        raise NotImplementedError


class BasicDictionizerDedictionizer(Dictionizer, Dedictionizer, ABC):
    def __init__(self, type_name: str, type_class):
        self.type_name = type_name
        self.type_class = type_class

    def is_dictable(self, key: str, value: Any) -> bool:
        return isinstance(value, self.type_class)

    def is_dedictable(self, key: str, value: Any) -> bool:
        return key == self.type_name or key.endswith(f"_{self.type_name}")

    def add_type_indicator(self, key: str) -> str:
        if key == self.type_name or key.endswith(f"_{self.type_name}"):
            return key
        else:
            return f"{key}_{self.type_name}"

    def remove_type_indicator(self, key: str) -> str:
        if key == self.type_name:
            return key
        else:
            return key[: -len(self.type_name) - 1]


class SerializeWgs84(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("wgs84", Point)

    def to_dict(self, key: str, value: Point) -> Tuple[str, dict]:
        key = self.add_type_indicator(key)
        return key, {"lat": value.wgs84().lat, "lon": value.wgs84().lon}

    def from_dict(self, key: str, value: dict[str, str]) -> Tuple[str, Optional[Point]]:
        key = self.remove_type_indicator(key)
        if "lat" in value and "lon" in value:
            return key, WgsPoint(lat=float(value["lat"]), lon=float(value["lon"]))
        else:
            return key, None


class SerializeAngle(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("degrees", Angle)

    def to_dict(self, key: str, value: Angle) -> Tuple[str, str]:
        key = self.add_type_indicator(key)
        return key, str(value.degrees)

    def from_dict(self, key: str, value: str) -> Tuple[str, Angle]:
        key = self.remove_type_indicator(key)
        return key, Angle(float(value))


class SerializeTimestamp(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("timestamp", datetime.datetime)

    def to_dict(self, key: str, value: datetime.datetime) -> Tuple[str, str]:
        key = self.add_type_indicator(key)
        return key, value.isoformat()

    def from_dict(self, key: str, value: str) -> Tuple[str, datetime.datetime]:
        key = self.remove_type_indicator(key)
        return key, datetime.datetime.fromisoformat(value)


class SerializeDuration(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("timedelta", datetime.timedelta)

    def to_dict(self, key: str, value: datetime.timedelta) -> Tuple[str, str]:
        key = self.add_type_indicator(key)
        return key, str(value.total_seconds())

    def from_dict(self, key: str, value: str) -> Tuple[str, datetime.timedelta]:
        key = self.remove_type_indicator(key)
        return key, datetime.timedelta(seconds=float(value))


class SerializeCell(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("cell", CellIdentity)

    def to_dict(self, key: str, value: CellIdentity) -> Tuple[str, dict[str, str]]:
        key = self.add_type_indicator(key)
        return key, {"radio": value.radio, "identifier": value.unique_identifier}

    def from_dict(
        self, key: str, value: dict[str, str]
    ) -> Tuple[str, Optional[CellIdentity]]:
        key = self.remove_type_indicator(key)
        if value.get("identifier") is None:
            return key, None
        else:
            return key, CellIdentity.parse(value.get("identifier"), value.get("radio"))


class SerializeLocationInfo(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("geo", LocationInfo)

    def to_dict(self, key: str, value: LocationInfo) -> Tuple[str, dict[str, str]]:
        key = self.add_type_indicator(key)
        return key, value

    def from_dict(self, key: str, value: dict[str, str]) -> Tuple[str, LocationInfo]:
        key = self.remove_type_indicator(key)
        return key, LocationInfo(**value)


class SerializeInteger(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("int", int)

    def is_dictable(self, key: str, value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def to_dict(self, key: str, value: int) -> Tuple[str, int]:
        key = self.add_type_indicator(key)
        return key, value

    def from_dict(self, key: str, value: str) -> Tuple[str, int]:
        key = self.remove_type_indicator(key)
        return key, int(value)


class SerializeFloat(BasicDictionizerDedictionizer):
    def __init__(self):
        super().__init__("float", float)

    def is_dictable(self, key: str, value: Any) -> bool:
        return isinstance(value, float)

    def to_dict(self, key: str, value: float) -> Tuple[str, float]:
        key = self.add_type_indicator(key)
        return key, value

    def from_dict(self, key: str, value: str) -> Tuple[str, float]:
        key = self.remove_type_indicator(key)
        return key, float(value)


class SerializeBoolean(Dictionizer, Dedictionizer):
    def is_dictable(self, key: str, value: Any) -> bool:
        return key.split("_")[0] in {"is", "has"}

    def is_dedictable(self, key: str, value: Any) -> bool:
        return self.is_dictable(key, value)

    def to_dict(self, key: str, value: bool) -> Tuple[str, bool]:
        return key, value

    def from_dict(self, key: str, value: str) -> Tuple[str, bool]:
        if isinstance(value, int):
            return key, bool(value)  # already deserialized
        else:
            return key, value.lower() in {"1", "true"}


CELL_MEASUREMENT_SERIALIZERS = [
    SerializeWgs84(),
    SerializeAngle(),
    SerializeTimestamp(),
    SerializeDuration(),
    SerializeCell(),
    SerializeLocationInfo(),
    SerializeInteger(),
    SerializeFloat(),
    SerializeBoolean(),
]


def _deserialize_property(
    deserializers: Iterable[Dedictionizer], key: str, value: Any
) -> Tuple[str, Any]:
    for deserializer in deserializers:
        if deserializer.is_dedictable(key, value):
            return deserializer.from_dict(key, value)

    return key, value


def _serialize_property(
    serializers: Iterable[Dictionizer], key: str, value: Any, blacklist_types: set
) -> Tuple[str, Any]:
    for serializer in serializers:
        if serializer.is_dictable(key, value) and serializer not in blacklist_types:
            return serializer.to_dict(key, value)

    assert (
        value is None or isinstance(value, str) or isinstance(value, dict)
    ), f"unable to serialize value '{key}' of type: {type(value)}"
    return key, value


def serialize_cell_measurement(
    measurement: CellMeasurement, blacklist_types: set = None
) -> dict[str, Any]:
    if blacklist_types is None:
        blacklist_types = set()
    return _collapse_dict(
        measurement.as_dict(), CELL_MEASUREMENT_SERIALIZERS, blacklist_types
    )


def deserialize_cell_measurement(measurement: dict[str, Any]) -> CellMeasurement:
    return CellMeasurement(**_expand_dict(measurement, CELL_MEASUREMENT_SERIALIZERS))


def _expand_dict(fields: dict, deserializers: Iterable[Dedictionizer]) -> dict:
    extra = Properties()
    keys = list(fields.keys())
    top_level = set([key.split(".")[0] for key in keys])
    for item in top_level:
        if item in fields:
            # we have an atomic value, probably a string value
            # serialize it if it is indeed a string;
            # otherwise take the value as-is, as it is already deserialized
            if fields[item] is not None:
                item, value = _deserialize_property(deserializers, item, fields[item])
            else:
                value = fields[item]
        else:
            # we have a composite value, probably a dictionary, but it is collapsed
            # find all fields in the hierarchy below this value
            properties = Properties(
                (key[len(item) + 1 :], value)
                for key, value in fields.items()
                if key.startswith(f"{item}.")
            )
            item, value = _deserialize_property(
                deserializers,
                item,
                _expand_dict(properties, deserializers),
            )

        extra[item] = value

    return extra


def _collapse_dict(d: dict, serializers: Iterable[Dictionizer], blacklist_types: set):
    result = {}
    for key, value in d.items():
        if value is not None:
            key, value = _serialize_property(serializers, key, value, blacklist_types)
        if isinstance(value, dict):
            result.update(
                (f"{key}.{k}", v)
                for k, v in _collapse_dict(value, serializers, blacklist_types).items()
            )
        else:
            result[key] = value

    return result


def _write_dictable_to_csv(
    f: Any,
    items: Iterable[Any],
    serializers: Iterable[Dictionizer],
    blacklist_types: set,
):
    if isinstance(f, str):
        if f == "-":
            return _write_dictable_to_csv(
                sys.stdout, items, serializers, blacklist_types
            )
        else:
            with open(f, "wt") as flo:
                return _write_dictable_to_csv(flo, items, serializers, blacklist_types)
    else:
        writer = csv.writer(f)
        labels = None
        for item in items:
            fields = _collapse_dict(item.as_dict(), serializers, blacklist_types)
            if labels is None:
                labels = sorted(fields.keys())
                writer.writerow(labels)
            writer.writerow(
                [fields[label] if label in fields else "" for label in labels]
            )


def _read_dictable_from_csv(
    file: Any,
    resource_name: Optional[str],
    deserializers: Iterable[Dedictionizer],
    create_item: Callable,
) -> Iterator:
    if isinstance(file, str):
        if resource_name is None:
            resource_name = file

        if file == "-":
            yield from _read_dictable_from_csv(
                sys.stdin, resource_name, deserializers, create_item
            )
        else:
            with open(file, "rt") as flo:
                yield from _read_dictable_from_csv(
                    flo, resource_name, deserializers, create_item
                )
    else:
        if resource_name is None:
            resource_name = ""

        reader = csv.reader(file)
        header = next(reader)
        for rownum, row in enumerate(reader):
            row = [v if v != "" else None for v in row]
            record = _expand_dict(dict(zip(header, row)), deserializers)
            record["id"] = f"{resource_name}:{rownum+2}"
            yield create_item(**record)


def read_measurements_from_csv(
    file: Any, resource_name: Optional[str] = None
) -> CellMeasurementSet:
    """
    Reads measurements from comma-separated-values.

    @param file: a filename (`str`) or file-like object or the special filename `'-'` (standard input).
    @param resource_name: human-readable name (`str`)  to refer to the file (defaults to filename)
    @return: a `CellMeasurementSet`
    """
    return CellMeasurementSet.from_measurements(
        _read_dictable_from_csv(
            file,
            resource_name,
            CELL_MEASUREMENT_SERIALIZERS,
            CellMeasurement,
        )
    )


def write_measurements_to_csv(
    file: Any,
    measurements: Iterable[CellMeasurement],
    blacklist_types: Optional[set] = None,
):
    """
    Writes measurements as comma-separated-values.

    This method optionally receives a set of value types to be blacklisted. This prevents these types of data to be
    written.

    @param file: a filename (`str`) or file-like object or the special filename `'-'` (standard input).
    @param measurements: the measurements to be written (iterable).
    @param blacklist_types: value types to be blacklisted.
    """
    if blacklist_types is None:
        blacklist_types = set()
    _write_dictable_to_csv(
        file,
        measurements,
        CELL_MEASUREMENT_SERIALIZERS,
        blacklist_types=blacklist_types,
    )
