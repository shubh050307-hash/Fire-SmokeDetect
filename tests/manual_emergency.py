import os

from dotenv import load_dotenv

from fire_agent import FireManagementAgent


load_dotenv()


def run_test():
    print("STARTING EMERGENCY WORKFLOW TEST...")

    agent = FireManagementAgent()

    detection_data = {
        "zone_id": 2,
        "coordinates": {"x": 450.5, "y": 320.2},
        "segment_area_pixels": 1250.0,
        "skip_email": os.getenv("AUTO_ALERTS_ENABLED", "false").lower() != "true",
    }

    print("\n[AI REASONING IN PROGRESS...]")
    result = agent.reason(detection_data)

    print("\n" + "=" * 50)
    print("TEST COMPLETE")
    print("=" * 50)

    if result and result.get("incident_log"):
        print(f"Incident Logged: {result['incident_log'].get('action', 'N/A')}")
    else:
        print("No incident was logged by the agent. Check the backend logs above.")


if __name__ == "__main__":
    run_test()
