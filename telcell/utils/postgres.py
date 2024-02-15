import logging
from typing import Optional, Mapping

import psycopg2

LOG = logging.getLogger(__name__)

DEFAULT_SEARCH_PATH = ["public", "contrib"]


def pgconnect(
    credentials: Mapping,
    schema: Optional[str] = None,
    use_wrapper: bool = True,
    statement_timeout=None,
):
    credentials = dict(credentials.items())
    search_path = (
        [schema] + DEFAULT_SEARCH_PATH if schema is not None else DEFAULT_SEARCH_PATH
    )
    credentials["options"] = f'-c search_path={",".join(search_path)}'
    if statement_timeout is not None:
        credentials["options"] += f" -c statement_timeout={statement_timeout}"
    con = psycopg2.connect(**credentials)

    if use_wrapper:
        con = Connection(con)

    return con


class Cursor:
    def __init__(self, con, commit_on_close, **kw):
        assert con is not None
        self.connection = con
        self.commit_on_close = commit_on_close
        self._cur = self.connection._con.cursor(**kw)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def execute(self, q, args=None):
        # LOG.debug('query: {q}; args: {args}'.format(q=q, args=args))
        try:
            self._cur.execute(q, args)
        except Exception as e:
            LOG.warning(
                "query failed: {q}; args: {args}; error: {e}".format(
                    q=q, args=args, e=e
                )
            )
            raise

    def __getattr__(self, name):
        return eval("self._cur.%s" % name)

    def __iter__(self):
        return self._cur.__iter__()

    def close(self):
        assert self._cur is not None, "already closed"
        if self.commit_on_close:
            LOG.debug("commit changes to database")
            self.connection.commit()
        self._cur.close()
        self._cur = None


class Connection:
    def __init__(self, con, autocommit=False):
        self._con = con
        self._autocommit = autocommit
        con.set_client_encoding("UTF8")

    def cursor(self, autocommit=None, **kw):
        return Cursor(
            self, autocommit if autocommit is not None else self._autocommit, **kw
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def commit(self):
        self._con.commit()

    def seed(self, seed: float):
        with self._con.cursor() as cur:
            cur.execute("SELECT setseed(%s)", (seed,))

    def close(self):
        assert self._con is not None, "already closed"
        if self._autocommit:
            self.commit()
        self._con.close()
        self._con = None

    def create_engine(self):
        from sqlalchemy import create_engine

        return create_engine("postgresql://", creator=lambda: self._con)


def drop_schema(con, schema):
    """
    Create or delete a schema in the db connected to by con

    :param con: database connection
    :param schema: schema name
    """
    with con.cursor() as cur:
        cur.execute("DROP SCHEMA IF EXISTS %s CASCADE" % schema)
    con.commit()


def create_schema(con, schema):
    """
    Create or delete a schema in the db connected to by con

    :param con: database connection
    :param schema: schema name
    """
    with con.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS %s" % schema)
    con.commit()
