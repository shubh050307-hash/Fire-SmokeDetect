import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure the root directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fire_backend import app
from fire_agent import get_recipient_emails

class TestParticipantsAPI(unittest.TestCase):
    def test_participants_crud_flow(self):
        print("\n🧪 Testing Participants CRUD Flow...")
        with TestClient(app) as client:
            # 1. Fetch current participants
            response = client.get("/api/participants")
            self.assertEqual(response.status_code, 200)
            initial_list = response.json()
            print(f"   Initial participants count: {len(initial_list)}")
            
            # 2. Add a new participant
            test_email = "test_participant_99@firewatch.ai"
            new_participant = {
                "name": "Test Participant 99",
                "email": test_email,
                "role": "Operator"
            }
            
            # Clean up if already exists from previous runs
            for p in initial_list:
                if p["email"] == test_email:
                    client.delete(f"/api/participants/{p['participant_id']}")
            
            response = client.post("/api/participants", json=new_participant)
            self.assertEqual(response.status_code, 200)
            added = response.json()
            self.assertEqual(added["name"], "Test Participant 99")
            self.assertEqual(added["email"], test_email)
            self.assertEqual(added["role"], "Operator")
            self.assertTrue(added["is_active"])
            participant_id = added["participant_id"]
            print(f"   ✅ Successfully added participant: {test_email} (ID: {participant_id})")

            # Verify dynamic loader fetches it
            emails = get_recipient_emails()
            self.assertIn(test_email, emails)
            print(f"   ✅ Dynamic agent receiver list includes: {test_email}")

            # 3. Toggle active status to false
            response = client.patch(f"/api/participants/{participant_id}/status", params={"is_active": False})
            self.assertEqual(response.status_code, 200)
            status_res = response.json()
            self.assertFalse(status_res["is_active"])
            print(f"   ✅ Successfully deactivated participant: {test_email}")

            # Verify dynamic loader does NOT fetch it when inactive
            emails = get_recipient_emails()
            self.assertNotIn(test_email, emails)
            print(f"   ✅ Dynamic agent receiver list correctly excludes deactivated email.")

            # 4. Toggle active status back to true
            response = client.patch(f"/api/participants/{participant_id}/status", params={"is_active": True})
            self.assertEqual(response.status_code, 200)
            status_res = response.json()
            self.assertTrue(status_res["is_active"])
            print(f"   ✅ Successfully reactivated participant: {test_email}")

            # 5. Delete participant
            response = client.delete(f"/api/participants/{participant_id}")
            self.assertEqual(response.status_code, 200)
            print(f"   ✅ Successfully deleted participant (ID: {participant_id})")

            # Verify dynamic loader does NOT fetch it after delete
            emails = get_recipient_emails()
            self.assertNotIn(test_email, emails)
            print(f"   ✅ Dynamic agent receiver list correctly excludes deleted email.")

if __name__ == "__main__":
    unittest.main()
