import argparse
import logging
import os
from logging.handlers import RotatingFileHandler

import confidence

from telcell.utils import postgres

DEFAULT_LOGLEVEL = logging.WARNING


def load_config():
    return confidence.load_name("cellsite", "../../cellsite", "local", "../../local")


def build_argparser(cfg, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument(
        "--schema",
        help=f"database schema (default: {cfg.database.schema})",
        default=cfg.database.schema,
    )
    parser.add_argument(
        "--drop-schema",
        action="store_true",
        help="drop schema before doing anything else",
    )
    parser.add_argument("-v", help="increases verbosity", action="count", default=0)
    parser.add_argument("-q", help="decreases verbosity", action="count", default=0)
    return parser


def get_database_connection(config_path: str, drop_schema: bool = False):
    cfg = confidence.loadf(*config_path.split(","))
    credentials = cfg.database.credentials
    schema = cfg.database.schema

    with postgres.pgconnect(credentials=credentials) as con:
        primary_schema = schema.split(",")[0]
        if drop_schema:
            postgres.drop_schema(con, primary_schema)
        postgres.create_schema(con, primary_schema)

    return postgres.pgconnect(credentials=credentials, schema=schema, use_wrapper=True)


def setup_logging(path, verbosity_offset):
    loglevel = max(
        logging.DEBUG, min(logging.CRITICAL, DEFAULT_LOGLEVEL - verbosity_offset * 10)
    )

    # setup formatter
    log_format = "[%(asctime)-15s %(levelname)s] %(name)s: %(message)s"
    fmt = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(loglevel)
    logging.getLogger().addHandler(ch)

    # setup a file handler
    if os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = RotatingFileHandler(path, mode="a", backupCount=3)
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logging.getLogger().addHandler(fh)

    logging.getLogger("").setLevel(logging.DEBUG)
