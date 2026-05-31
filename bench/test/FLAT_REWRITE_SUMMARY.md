# Flat SQL rewrite summary
Converted queries in manifest_flat.json: 58
Unsupported non-future queries: 6

## Unsupported
- `00_health/health_02_count_customer.sql`: flat table is fact-rooted; query does not reference lineorder
- `00_health/health_03_count_part.sql`: flat table is fact-rooted; query does not reference lineorder
- `00_health/health_04_count_supplier.sql`: flat table is fact-rooted; query does not reference lineorder
- `00_health/health_05_count_ddate.sql`: flat table is fact-rooted; query does not reference lineorder
- `02_projection/proj_05_select_star_small_dimension.sql`: flat table is fact-rooted; query does not reference lineorder
- `08_custom/custom1.sql`: flat table stores ddate attributes for lo_orderdate only, not for lo_commitdate
