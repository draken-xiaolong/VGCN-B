# Data license

## OpenStreetMap-derived files

The files matching `*-latest-free.shp-gis_osm_*.geojson` in

- `convertToGeoJson/GeoJson/TrainingSet/`
- `convertToGeoJson/GeoJson/TestSet/`
- `zNC-Test/vector-data-geojson/`

contain map data © OpenStreetMap contributors, obtained from the Geofabrik free
shapefile extracts for Shanghai and Tianjin (https://download.geofabrik.de) and
converted to GeoJSON without editing the geometries.

These files are made available under the Open Database License (ODbL) 1.0:
https://opendatacommons.org/licenses/odbl/1-0/ . Any rights in individual
contents of the database are licensed under the Database Contents License:
https://opendatacommons.org/licenses/dbcl/1-0/ . See also
https://www.openstreetmap.org/copyright .

If you publicly use or redistribute these files or a database derived from them,
you must attribute OpenStreetMap contributors and keep the derived database
under the ODbL.

## National 1:1,000,000 public basic geographic data

The `H50-*` and `H51-*` files are not included in this repository. They were
obtained from the National Geographic Information Service Platform of China
(全国地理信息资源目录服务系统, https://www.webmap.cn), whose user agreement does
not permit redistribution. Download them from the portal; `input_manifest.json`
lists the expected file names and SHA-256 hashes.

## Code

The license of the source code is separate from the data licenses above.
