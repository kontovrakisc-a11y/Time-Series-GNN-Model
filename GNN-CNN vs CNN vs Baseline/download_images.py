import os
import requests
import pandas as pd
from PIL import Image
from io import BytesIO

def download_satellite_image(lat, lon, station_id, zoom=8, size=512, date="2024-07-01", save_dir="data/satellite"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    base_url = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
    delta = 0.5 / zoom 
    bbox = f"{lon - delta},{lat - delta},{lon + delta},{lat + delta}"

    params = {
        "SERVICE": "WMS", "VERSION": "1.3.0", "REQUEST": "GetMap",
        "LAYERS": "MODIS_Terra_CorrectedReflectance_TrueColor",
        "TIME": date, "CRS": "EPSG:4326", "BBOX": bbox,
        "WIDTH": size, "HEIGHT": size, "FORMAT": "image/jpeg"
    }

    try:
        response = requests.get(base_url, params=params, timeout=15)
        if "image" in response.headers.get("Content-Type", "").lower():
            img = Image.open(BytesIO(response.content))
            filename = f"station_{station_id}_satellite.jpg"
            img.save(os.path.join(save_dir, filename))
            return True
    except Exception as e:
        pass
    return False

def download_cimis_site_photos(station_id, save_dir="data/station_images"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    direction_map = {'North': 'Nt', 'South': 'St', 'East': 'Et', 'West': 'Wt'}
    padded_id = str(station_id).zfill(3)
    base_url = "https://cimis.water.ca.gov/App_Themes/Images/Stations"

    for direction_name, suffix in direction_map.items():
        img_name = f"{padded_id}{suffix}.jpg"
        url = f"{base_url}/{img_name}"
        
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                local_name = f"{station_id}_{direction_name}.jpg"
                with open(os.path.join(save_dir, local_name), 'wb') as f:
                    f.write(response.content)
        except Exception:
            pass

registry_df = pd.read_csv("data/station_registry.csv")
for _, row in registry_df.iterrows():
    download_satellite_image(row['Lat'], row['Lon'], row['ID'])
    download_cimis_site_photos(row['ID'])
print("Images downloaded!")
