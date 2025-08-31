import folium

# --- Map Configuration ---
# Coordinates for the center of SUST campus (Shahjalal University of Science and Technology)
sust_campus_center = [24.9188, 91.8315]

# The zoom level to show the campus in detail (16-19 is usually good for a campus)
zoom_level = 100

# --- Create the Base Map ---
# The map uses OpenStreetMap by default, which has excellent road and building details.
sust_map = folium.Map(
    location=sust_campus_center, 
    zoom_start=zoom_level,
    tiles="OpenStreetMap"
)

# --- Add Campus Landmarks ---
# Add a marker for the Main Gate
folium.Marker(
    location=[24.9184, 91.8340],
    tooltip="Main Gate",
    popup="<b>SUST Main Gate</b>",
    icon=folium.Icon(color='green', icon='info-sign')
).add_to(sust_map)

# Add a marker for the Central Library
folium.Marker(
    location=[24.9188, 91.8298],
    tooltip="Central Library",
    popup="<b>Central Library</b>",
    icon=folium.Icon(color='blue', icon='book')
).add_to(sust_map)

# Add a marker for the University Auditorium
folium.Marker(
    location=[24.9168, 91.8286],
    tooltip="University Auditorium",
    popup="<b>Auditorium</b>",
    icon=folium.Icon(color='purple', icon='music')
).add_to(sust_map)

# --- Add a path for a main campus road (example) ---
main_road_path = [
    [24.9185, 91.8340],
    [24.9175, 91.8315],
    [24.9168, 91.8290],
    [24.9165, 91.8295]
]

# folium.PolyLine(
#     locations=main_road_path,
#     color='gray',
#     weight=6,
#     opacity=0.7,
#     tooltip="Main Campus Road"
# ).add_to(sust_map)

# --- Save the Map ---
map_file_name = "sust_campus_map.html"
sust_map.save(map_file_name)

print(f"Interactive map of SUST campus saved to {map_file_name}")
print("Open this file in your web browser to view the map.")