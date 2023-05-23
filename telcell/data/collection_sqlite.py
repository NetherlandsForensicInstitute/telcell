from __future__ import annotations

import datetime
import logging
import random
import sqlite3
import string
from abc import ABC, abstractmethod
from contextlib import closing
from typing import Any, Iterable, Sized, Optional, Set, Iterator, Tuple, Sequence

from .item import Item
from .serialization import (
    SerializeBoolean,
    SerializeInteger,
    SerializeFloat,
    SerializeAngle,
    SerializeTimestamp,
    SerializeDuration,
)


SQL_TYPES = [
    (SerializeBoolean(), "INT"),
    (SerializeInteger(), "INT"),
    (SerializeFloat(), "FLOAT"),
    (SerializeAngle(), "FLOAT"),
    (SerializeTimestamp(), "DATETIME"),
    (SerializeDuration(), "FLOAT"),
]

LOG = logging.getLogger(__name__)


def get_sql_type(key: str, value: Any) -> str:
    for serializer, type_name in SQL_TYPES:
        if serializer.is_dedictable(key.split(".")[-1], value):
            return type_name

    return "VARCHAR"


def _get_fields_for_item(item):
    fields = []
    for key, value in item.items():
        fields.append((key, get_sql_type(key, value)))
    return dict(fields)


def _sql_safe_name(name):
    return f'"{name}"'


def escape_sql(s: str) -> str:
    return s.replace("'", "''")


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    elif isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    elif isinstance(value, int):
        return str(value)
    elif isinstance(value, datetime.datetime):
        return f"'{value.isoformat()}'"
    elif isinstance(value, datetime.timedelta):
        return str(value.total_seconds())
    else:
        return f"'{escape_sql(str(value))}'"


class SqliteCollection(Iterable[Item], Sized, ABC):
    def __init__(
        self,
        items: Optional[Iterable[Item]] = None,
        con=None,
        table_name: str = "item",
        parent: SqliteCollection = None,
    ):
        self._con = con if con is not None else sqlite3.connect(":memory:")
        self._table_name = table_name
        self._parent = parent

        self._fields = self._find_fields()
        if items is not None:
            for item in items:
                self.add(item)

    def __del__(self):
        if self._parent is not None:
            with closing(self._con.cursor()) as cur:
                cur.execute(
                    f"""
                    DROP VIEW {self._table_name}
                """
                )

    @property
    def _root_name(self):
        return self._table_name if self._parent is None else self._parent._root_name

    def _get_columns(self):
        return [
            f"{_sql_safe_name(field_name)} {field_type}"
            for field_name, field_type in self._fields.items()
        ]

    def _create_table(self, fields: dict[str, str]):
        self._fields = fields
        LOG.debug(
            f"creating table {self._table_name} with fields: {'; '.join(fields.keys())}"
        )
        with closing(self._con.cursor()) as cur:
            cur.execute(
                f"""
                CREATE TABLE {self._table_name} ({','.join(self._get_columns())})
            """
            )

    def _create_index(self, field: str):
        if self._fields is None:
            return
        assert self._root_name is not None, "illegal state: root table is unknown"
        LOG.debug(f"creating index for {field} on {self._root_name}")
        with closing(self._con.cursor()) as cur:
            index_name = _sql_safe_name(f"{self._root_name}_{field}")
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS {index_name} ON {self._root_name}("{field}")'
            )

    def _find_fields(self) -> dict[str, str]:
        with closing(self._con.cursor()) as cur:
            cur.execute(f"PRAGMA table_info({self._table_name})")
            fields = dict(
                (name, type)
                for cid, name, type, notnull, dflt_value, pk in cur.fetchall()
            )
            return None if len(fields) == 0 else fields

    def add(self, item: Item):
        item = self.serialize_item(item)
        if self._fields is None:
            self._create_table(_get_fields_for_item(item))
        else:
            fields = _get_fields_for_item(item)
            for field_name in fields:
                if field_name not in self._fields:
                    # the item has an unknown field -> add the new field to all existing items
                    with closing(self._con.cursor()) as cur:
                        parent_field = ".".join(field_name.split(".")[:-1])
                        if parent_field in self._fields:
                            # we have a conflict -> rename the original field
                            LOG.debug(
                                f"table {self._table_name}: renaming field {parent_field} to {parent_field}.value"
                            )
                            cur.execute(
                                f'ALTER TABLE {self._table_name} RENAME COLUMN "{parent_field}" TO "{parent_field}.value"'
                            )
                            self._fields[f"{parent_field}.value"] = self._fields[
                                parent_field
                            ]

                            del self._fields[parent_field]

                        self._fields[field_name] = fields[field_name]
                        LOG.debug(f"{self._table_name}: adding column {field_name}")
                        cur.execute(
                            f'ALTER TABLE {self._table_name} ADD COLUMN "{field_name}" {fields[field_name]}'
                        )

        with closing(self._con.cursor()) as cur:
            q = f"""
                INSERT INTO {self._table_name}({','.join(_sql_safe_name(field) for field in self._fields)}) VALUES({','.join([sql_literal(item.get(field, None)) for field in self._fields])})
            """
            qargs = []
            cur.execute(q, qargs)

    def __iter__(self) -> Iterator[Item]:
        if self._fields is None:
            return

        qselect = ",".join(_sql_safe_name(field) for field in self._fields)
        q, qargs = self._build_query(qselect)

        with closing(self._con.cursor()) as cur:
            cur.execute(q, qargs)
            for item in cur:
                item = dict(zip(self._fields, item))
                item = self.deserialize_item(item)
                yield item

    def __len__(self):
        if self._fields is None:
            return 0

        qselect = "COUNT(*)"
        q, qargs = self._build_query(qselect)
        with closing(self._con.cursor()) as cur:
            cur.execute(q, qargs)
            return cur.fetchone()[0]

    def __add__(self, other):
        items = self.create_collection()
        for m in self:
            items.add(m)
        for m in other:
            items.add(m)
        return items

    def __iadd__(self, other):
        for m in other:
            self.add(m)
        return self

    def get_unique_values(self, *field_names: str) -> Set[str] | Set[Tuple]:
        if self._fields is None:
            return set()

        for field_name in field_names:
            assert (
                field_name in self._fields
            ), f"collection items have no `{field_name}` attribute (fields available: {self._fields})"
            self._create_index(field_name)
        columns = [f'"{field_name}"' for field_name in field_names]
        qselect = f'DISTINCT {", ".join(columns)}'
        q, qargs = self._build_query(qselect)
        with closing(self._con.cursor()) as cur:
            cur.execute(q, qargs)
            if len(field_names) == 1:
                return set([row[0] for row in cur])
            else:
                return set(cur)

    @abstractmethod
    def create_collection(
        self,
        items: Optional[Iterable[Item]] = None,
        sqlite_args: dict = None,
    ) -> SqliteCollection:
        raise NotImplementedError

    @abstractmethod
    def serialize_item(self, item: Item) -> dict:
        raise NotImplementedError

    @abstractmethod
    def deserialize_item(self, item: dict[str, str]) -> Item:
        raise NotImplementedError

    def select_by_value(self, **constraints):
        for field_name in constraints:
            self._create_index(field_name)
        return self._create_view(equals_constraints=constraints)

    def select_by_range(self, **constraints):
        for field_name in constraints:
            self._create_index(field_name)
        return self._create_view(range_constraints=constraints)

    def select_by_values(self, **constraints):
        for field_name in constraints:
            self._create_index(field_name)
        return self._create_view(in_collection_constraints=constraints)

    def limit(self, count: int) -> SqliteCollection:
        return self._create_view(limit=count)

    def sort_by(self, *keys: str) -> SqliteCollection:
        for field_name in keys:
            if field_name in self._fields:
                self._create_index(field_name)
        return self._create_view(sort_keys=keys)

    def with_value(self, **values) -> SqliteCollection:
        return self.create_collection(item.with_value(**values) for item in self)

    def _create_view(
        self,
        equals_constraints: dict[str, Any] = None,
        in_collection_constraints: dict[str, Iterable] = None,
        range_constraints: dict[str, Tuple[Any, Any]] = None,
        sort_keys: Optional[Sequence[str]] = None,
        limit: int = None,
    ) -> SqliteCollection:
        if self._fields is None:
            return self.create_collection()

        view_name = "".join(random.choice(string.ascii_lowercase) for _ in range(10))
        columns = [_sql_safe_name(field_name) for field_name in self._fields.keys()]
        q, qargs = self._build_query(
            ",".join(columns),
            equals_constraints=equals_constraints,
            in_collection_constraints=in_collection_constraints,
            range_constraints=range_constraints,
            limit=limit,
            sort_keys=sort_keys,
        )
        with closing(self._con.cursor()) as cur:
            q = f"""
                CREATE TEMPORARY VIEW {view_name} ({','.join(f"{col}" for col in columns)})
                AS
                {q}
            """
            cur.execute(q, qargs)
            sqlite_args = {
                "con": self._con,
                "table_name": view_name,
                "parent": self,
            }
            return self.create_collection(sqlite_args=sqlite_args)

    def _build_query(
        self,
        qselect: str,
        equals_constraints: dict[str, str] = None,
        in_collection_constraints: dict[str, Sequence] = None,
        range_constraints: dict[str, Tuple[Any, Any]] = None,
        sort_keys: Optional[Sequence[str]] = None,
        limit: int = None,
    ) -> Tuple[str, Sequence]:
        qwhere = []
        qargs = []

        if equals_constraints is not None:
            for key, value in equals_constraints.items():
                if value is None:
                    qwhere.append(f'"{key}" is NULL')
                else:
                    qwhere.append(f'"{key}" = {sql_literal(value)}')

        if in_collection_constraints is not None:
            for key, value in in_collection_constraints.items():
                values = [sql_literal(s) for s in value]
                qwhere.append(f"\"{key}\" in ({','.join(values)})")

        if range_constraints is not None:
            for key, (value_from, value_to) in range_constraints.items():
                if value_from is not None:
                    qwhere.append(f'"{key}" >= {sql_literal(value_from)}')
                if value_to is not None:
                    qwhere.append(f'"{key}" < {sql_literal(value_to)}')

        if len(qwhere) == 0:
            qwhere.append("1")

        qorder = ""
        if sort_keys is not None:
            sort_key = [
                (f'"{key}"' if key in self._fields else key) for key in sort_keys
            ]
            qorder = f"ORDER BY {', '.join(sort_key)}"

        qlimit = ""
        if limit is not None:
            qlimit = f"LIMIT {limit}"

        q = f"""
            SELECT {qselect}
            FROM {self._table_name}
            WHERE {' AND '.join(qwhere)}
            {qorder}
            {qlimit}
        """

        return q, qargs
