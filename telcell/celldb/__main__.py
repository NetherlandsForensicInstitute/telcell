import sys
from functools import partial
from typing import Sequence

import click
from tqdm import tqdm

from telcell.celldb import pgdatabase
from telcell.utils import script_helper


def create_antenna_light_table(con):
    pgdatabase.create_table(con)
    tablename = "antenna_light"
    with con.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {tablename}(date_start, date_end, radio, mcc, mnc, lac, ci, eci, rd, azimuth)
            SELECT start, "end",
                CASE
                    WHEN radio = '2G' THEN 'GSM'
                    WHEN radio = '3G' THEN 'UMTS'
                    WHEN radio = '4G' THEN 'LTE'
                    WHEN radio = '5G' THEN 'NR'
                    ELSE radio
                END radio,
                204, mnc,
                lac,
                CASE
                    WHEN radio = '4G' THEN NULL
                    ELSE cell
                END ci,
                CASE
                    WHEN eci is NOT NULL THEN eci
                    WHEN radio = '4G' AND cell is NOT NULL THEN cell
                    ELSE NULL
                END eci,
                'SRID=4326;POINT('||rdx||' '||rdy||')', azimuth
            FROM antenna
            WHERE cell is NOT NULL OR eci is NOT NULL
        """
        )


@click.group()
@click.option(
    "--config",
    metavar="FILE",
    required=True,
    help="Comma-separated list of YAML files with database credentials",
)
@click.option("--drop-schema", is_flag=True, help="Drop schema before doing anything else")
@click.pass_context
def cli(ctx, config: str, drop_schema: bool):
    ctx.ensure_object(dict)
    ctx.obj["CONFIG_PATH"] = config
    ctx.obj["DROP_SCHEMA"] = drop_schema
    script_helper.setup_logging("celldb.log", verbosity_offset=0)


@cli.command("import", help="import a CSV file into the database")
@click.argument(
    "filename",
    metavar="PATH",
    nargs=-1,
)
@click.pass_context
def csv_import(ctx, filename: Sequence[str]):
    with script_helper.get_database_connection(ctx.obj["CONFIG_PATH"]) as con:
        for file in filename:
            with open(file, "rt") as f:
                progress = partial(tqdm, desc=f"{file}: reading cells")
                pgdatabase.csv_import(con, f, progress)


@cli.command(help="export the database to CSV")
@click.pass_context
def export(ctx):
    with script_helper.get_database_connection(ctx.obj["CONFIG_PATH"]) as con:
        pgdatabase.csv_export(con, sys.stdout)


if __name__ == "__main__":
    cli()
