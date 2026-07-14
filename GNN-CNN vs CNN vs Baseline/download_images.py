import requests
import base64
import os
import time
import urllib3

# Disable unverified HTTPS warnings for clean console output
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_cimis_station_images():
    """
    Downloads weather station images (North, South, East, West) from the modern CIMIS API.
    The images are returned as raw base64 strings and are decoded and saved locally.
    """
    # The subscription key used by the modern CIMIS frontend to access the API
    headers = {
        'ocp-apim-subscription-key': '4fd07064a189429baf69cdefb98df8f3',
        'accept': 'application/json',
        'user-agent': 'Mozilla/5.0'
    }
    
    # Unified list of target CIMIS station IDs from Data_Collection.ipynb
    target_ids = [
        106, 103, 158, 83, 77, 144, 187, 157, 213, 178, 170,
        250, 226, 6, 139, 235, 212, 140, 247, 47, 242, 243, 248,
        254, 171, 191, 253, 104, 211,
        195, 13, 228, 227, 262, 131, 84, 70, 249, 71, 194, 206
    ]
    
    output_dir = 'cimis_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    print(f"Fetching images for {len(target_ids)} stations from production API...")
    found_count = 0
    
    for st_id in target_ids:
        # The production API endpoint for fetching station details
        url = f"https://et.water.ca.gov/ApiWeb/GetStationByStationId?stationId={st_id}"
        
        try:
            r = requests.get(url, headers=headers, verify=False, timeout=15)
            if r.status_code == 200:
                data = r.json()
                images = data.get('StationImages')
                
                # Check if the station has an array of images attached
                if images and isinstance(images, list):
                    for img_obj in images:
                        image_data = img_obj.get('Image', '')
                        desc = img_obj.get('Description', 'Unknown').replace('/', '_').replace(' ', '_')
                        
                        if image_data:
                            # If the image string has the standard data URI prefix, strip it off
                            if 'base64,' in image_data:
                                base64_str = image_data.split('base64,', 1)[1]
                            else:
                                base64_str = image_data
                                
                            try:
                                img_bytes = base64.b64decode(base64_str)
                                
                                # Default to jpg, simple check for png headers
                                ext = 'jpg'
                                if base64_str.startswith('iVBORw'):
                                    ext = 'png'
                                
                                file_path = os.path.join(output_dir, f"{st_id}_{desc}.{ext}")
                                with open(file_path, 'wb') as f:
                                    f.write(img_bytes)
                                    
                                found_count += 1
                                print(f"Saved image for station {st_id} ({desc})")
                                
                            except Exception as e:
                                print(f"Error decoding image for station {st_id}: {e}")
                else:
                    print(f"No images available in the database for station {st_id}")
            else:
                print(f"Failed to fetch station {st_id}: API returned status code {r.status_code}")
                
        except Exception as e:
            print(f"Connection error fetching station {st_id}: {e}")
            
        # Small delay to prevent API rate limiting
        time.sleep(0.2)
        
    print(f"\nTotal images downloaded and saved: {found_count}")

if __name__ == "__main__":
    download_cimis_station_images()
