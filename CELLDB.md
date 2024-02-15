Cell database management
========================

In many practical situations you will need a cell database which contains
information on the actual positions of the cell antennas, as well as other
meta data. The following assumes you have such a database in a readable CSV
format.

# Import cell database

Install Postgres and Postgis and remember credentials.

Postgis must be added to the database explicitly after installation:

```sql
CREATE EXTENSION postgis;
```

Create `celldb.yaml` from the template and insert the Postgres credentials.

```sh
cp celldb.yaml-example celldb.yaml
nano celldb.yaml
```

```sh
python -m telcell.celldb --config celldb.yaml import < celldb.csv
```

# Usage

```py
with script_helper.get_cell_database("celldb.yaml", on_duplicate=duplicate_policy.take_first) as db:
    # count the number of cells in the database
    print(len(db))

    # retrieve cell info
    my_cell = CellIdentity.create(radio="GSM", mcc=99, mnc=99, eci=123456)
    cellinfo = db.get(my_cell, date=datetime.datetime.now())
    if cellinfo is None:
        print("cell not found")
    else:
        print(cellinfo)

    # the search method returns a new `CellDatabase` object
    # for example, reduce the cell database to GSM only
    gsm_db = db.search(radio="GSM")
    print(f"found {len(gsm_db)} GSM cells")

    # find all cells within 5km of a point
    my_point = geopy.Point(52.1, 4.9)
    nearby_gsm_cells = gsm_db.search(coords=my_point, distance_limit_m=5000)
    print(f"{len(nearby_gsm_cells)} cells within 5km")
    for cellinfo in nearby_gsm_cells:
        print(cellinfo)
```
