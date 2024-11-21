Library for cell site analysis
==============================

### Import cell database

In many practical situations you will need a cell database which contains
information on the actual positions of the cell antennas, as well as other
meta data. The following assumes you have such a database in a readable CSV
format.

Install Postgres and Postgis and remember credentials.

Postgis must be added to the database explicitly after installation:

```sql
CREATE EXTENSION postgis;
```

Create `cellsite.yaml` from the template and insert the Postgres credentials.

```sh
cp cellsite.yaml-example cellsite.yaml
nano cellsite.yaml
```

```sh
python -m celldb --config cellsite.yaml import < celldb.csv
```

For more information, see `celldb` documentation.
