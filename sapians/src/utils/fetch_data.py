from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt

# Connect to Copernicus open hub (create free account)
api = SentinelAPI('username', 'password', 'https://apihub.copernicus.eu/apihub')

footprint = geojson_to_wkt(read_geojson('area.geojson'))
products = api.query(footprint, date=('20250601', '20250701'),
                     platformname='Sentinel-2', cloudcoverpercentage=(0, 10))
api.download_all(products, directory_path='../../data/')
