import requests

r = requests.get('https://archive-api.open-meteo.com/v1/archive', params={
    'latitude': 8.5574,
    'longitude': 76.8800,
    'start_date': '2025-03-01',
    'end_date': '2025-03-10',
    'hourly': 'temperature_2m',
    'timezone': 'Asia/Kolkata'
}, timeout=120)

print(r.status_code)
print(r.json())