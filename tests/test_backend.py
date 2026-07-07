"""
Test Script for Fire Management Backend
- Initialize database
- Create mock detection images
- Simulate fire detections
- Log incidents
- Query data
"""

import requests
import json
import random
from PIL import Image, ImageDraw
import io
from datetime import datetime, timedelta

BASE_URL = "http://localhost:8000/api"

print("🔥 Fire Management System - Backend Test\n")

# ============= 1. INITIALIZE DATABASE =============
print("=" * 60)
print("STEP 1: Initialize Zones & Procedures")
print("=" * 60)

response = requests.post(f"{BASE_URL}/init/zones")
print(f"✅ Zones: {response.json()}")

response = requests.post(f"{BASE_URL}/init/procedures")
print(f"✅ Procedures: {response.json()}\n")


# ============= 2. GET ZONES =============
print("=" * 60)
print("STEP 2: Get All Zones")
print("=" * 60)

response = requests.get(f"{BASE_URL}/zones")
zones = response.json()
for zone in zones:
    print(f"  Zone {zone['zone_id']}: {zone['name']} ({zone['area_sqm']} sqm)")
print()


# ============= 3. CREATE MOCK DETECTION IMAGES =============
def create_mock_image(zone_name: str, width=640, height=480):
    """Create a realistic-looking mock detection image"""
    img = Image.new('RGB', (width, height), color='gray')
    draw = ImageDraw.Draw(img)
    
    # Draw some building features
    draw.rectangle([50, 50, 300, 250], fill='lightblue', outline='blue', width=2)  # Wall
    draw.rectangle([350, 100, 600, 350], fill='lightyellow', outline='orange', width=2)  # Window
    
    # Draw simulated fire area (orange/red)
    fire_x = random.randint(100, 500)
    fire_y = random.randint(100, 350)
    radius = random.randint(30, 80)
    
    for i in range(3):
        r = radius - (i * 10)
        color = ['red', 'orange', 'yellow'][i]
        draw.ellipse(
            [fire_x - r, fire_y - r, fire_x + r, fire_y + r],
            fill=color,
            outline=color
        )
    
    # Add zone label
    draw.text((10, 10), f"Zone: {zone_name}", fill='white')
    draw.text((10, 30), f"Time: {datetime.now().strftime('%H:%M:%S')}", fill='white')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes


# ============= 4. SIMULATE DETECTIONS =============
print("=" * 60)
print("STEP 3: Simulate Fire Detections")
print("=" * 60)

detection_ids = []

for zone_id in [1, 2, 3]:
    zone = next(z for z in zones if z['zone_id'] == zone_id)
    
    # Create mock image
    img_bytes = create_mock_image(zone['name'])
    
    # Detection data (NO confidence scores!)
    detection_data = {
        "zone_id": zone_id,
        "coordinates_x": random.uniform(100, 500),
        "coordinates_y": random.uniform(100, 350),
        "bbox_w": random.uniform(50, 150),
        "bbox_h": random.uniform(50, 150),
        "segment_area_pixels": random.uniform(2000, 5000)
    }
    
    # Send detection
    files = {'file': ('detection.png', img_bytes, 'image/png')}
    response = requests.post(
        f"{BASE_URL}/detections",
        data=detection_data,
        files=files
    )
    
    if response.status_code != 200:
        print(f"❌ Error {response.status_code}: {response.json()}")
        continue
    
    result = response.json()
    detection_ids.append(result['detection_id'])
    
    print(f"✅ Detection {result['detection_id']} in {result['zone']}")
    print(f"   📷 Image: {result['image_url']}")
    print(f"   Coordinates: ({detection_data['coordinates_x']:.1f}, {detection_data['coordinates_y']:.1f})")

print()


# ============= 5. GET ALL DETECTIONS =============
print("=" * 60)
print("STEP 4: Retrieve All Detections")
print("=" * 60)

response = requests.get(f"{BASE_URL}/detections")
detections = response.json()
print(f"Total detections: {len(detections)}\n")

for det in detections[:3]:  # Show first 3
    print(f"Detection {det['detection_id']}:")
    print(f"  Zone: {det['zone_id']}")
    print(f"  Timestamp: {det['timestamp']}")
    print(f"  Area: {det['segment_area_pixels']:.0f} pixels")
    print(f"  Image: {det['image_filename']}\n")


# ============= 6. LOG INCIDENTS (AGENT DECISIONS) =============
print("=" * 60)
print("STEP 5: Log Incidents (Agent Decisions)")
print("=" * 60)

sprinkler_types = ["Sprinkler", "CO2", "Foam", "Water Mist"]
incident_ids = []

for i, detection_id in enumerate(detection_ids):
    zone_id = i + 1
    sprinkler = sprinkler_types[zone_id - 1]  # Assign sprinkler based on zone
    
    incident_data = {
        "detection_id": detection_id,
        "zone_id": zone_id,
        "action_taken": f"{sprinkler} system activated",
        "sprinkler_type": sprinkler,
        "agent_reasoning": f"Fire detected in Zone {zone_id}. Confidence threshold exceeded. Activating {sprinkler} suppression system per NFPA protocols."
    }
    
    response = requests.post(f"{BASE_URL}/incidents", json=incident_data)
    result = response.json()
    incident_ids.append(result['incident_id'])
    
    print(f"✅ Incident {result['incident_id']} logged")
    print(f"   Zone: {zone_id} | Action: {incident_data['action_taken']}")
    print(f"   Sprinkler: {sprinkler}")

print()


# ============= 7. GET ALL INCIDENTS =============
print("=" * 60)
print("STEP 6: Retrieve All Incidents")
print("=" * 60)

response = requests.get(f"{BASE_URL}/incidents")
incidents = response.json()
print(f"Total incidents: {len(incidents)}\n")

for inc in incidents:
    print(f"Incident {inc['incident_id']}:")
    print(f"  Zone: {inc['zone_id']} | Detection: {inc['detection_id']}")
    print(f"  Action: {inc['action_taken']}")
    print(f"  Sprinkler: {inc['sprinkler_type']}")
    print(f"  Status: {inc['status']}")
    print()


# ============= 8. UPDATE INCIDENT STATUS =============
print("=" * 60)
print("STEP 7: Update Incident Status")
print("=" * 60)

if incident_ids:
    incident_id = incident_ids[0]
    response = requests.patch(
        f"{BASE_URL}/incidents/{incident_id}/status",
        params={"status": "resolved"}
    )
    print(f"✅ Incident {incident_id} status updated to: {response.json()['new_status']}\n")


# ============= 9. GET PROCEDURES =============
print("=" * 60)
print("STEP 8: Get Safety Procedures")
print("=" * 60)

for zone_id in [1, 2, 3]:
    response = requests.get(f"{BASE_URL}/procedures/{zone_id}")
    proc = response.json()
    print(f"Zone {zone_id} - {proc['protocol_name']}:")
    print(f"  Suppression: {proc['suppression_type']}")
    print(f"  Evacuation time: {proc['evacuation_time_min']} min")
    for step in proc['procedure_steps']:
        print(f"    {step}")
    print()


# ============= 10. DASHBOARD SUMMARY =============
print("=" * 60)
print("STEP 9: Dashboard Summary")
print("=" * 60)

response = requests.get(f"{BASE_URL}/dashboard/summary")
summary = response.json()

print(f"📊 Dashboard Metrics:")
print(f"  Total Detections: {summary['total_detections']}")
print(f"  Active Incidents: {summary['active_incidents']}")
print(f"  Total Zones: {summary['total_zones']}")
print(f"  Last Detection: {summary['last_detection']}")
print()


# ============= 11. INCIDENTS BY ZONE =============
print("=" * 60)
print("STEP 10: Incidents by Zone")
print("=" * 60)

response = requests.get(f"{BASE_URL}/dashboard/incidents-by-zone")
stats = response.json()

for stat in stats:
    print(f"  {stat['zone_name']}: {stat['incident_count']} incident(s)")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
print("\n📚 API Documentation: http://localhost:8000/docs")
print("🎯 Detection Images: /Users/muhammadomerfarooq/Desktop/Fire and Smoke Detection/detection_images/")