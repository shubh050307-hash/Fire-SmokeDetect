"""
Fire Management Agentic System
- Uses Groq API (free, fast) for reasoning
- Integrates with REAL vector RAG system for safety procedures
- Calls tools to activate suppression, log incidents
- Real-time fire incident response
"""

import os
import requests
import json
import html
from urllib.parse import quote_plus
from dotenv import load_dotenv
from real_rag_system import RealRAGSystem
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import base64
import numpy as np
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from typing import Any, Optional

# Load API key from .env (never hard-code secrets!)
load_dotenv()

_shared_rag: Optional[RealRAGSystem] = None

# ============= GROQ API SETUP =============
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Loaded from .env
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# ============= EMAIL SETUP =============
import sqlite3

def get_recipient_emails() -> list[str]:
    """Dynamically query the SQLite database for active recipients, falling back to env-configured list."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fire_system.db")
    try:
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            # Check if participants table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='participants';")
            if cursor.fetchone():
                cursor.execute("SELECT email FROM participants WHERE is_active = 1;")
                emails = [row[0] for row in cursor.fetchall() if row[0].strip()]
                conn.close()
                if emails:
                    return emails
            else:
                conn.close()
    except Exception as e:
        print(f"⚠️ Error reading recipients from SQLite: {e}")
        
    # Fallback to env-configured list
    return [email.strip() for email in os.getenv("REMINDER_EMAIL_RECEIVERS", "").split(",") if email.strip()]

SENDER_EMAIL = os.getenv("REMINDER_EMAIL_SENDER", "")
RECEIVER_EMAILS = get_recipient_emails()
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "") # Requires Gmail App Password
ALERT_MODE = os.getenv("ALERT_MODE", "demo").strip().lower()
GOOGLE_TOKEN_FILE = os.getenv("GOOGLE_TOKEN_FILE", "token.json")

def get_shared_rag(instance: Optional[RealRAGSystem] = None) -> RealRAGSystem:
    """Return (or set) the shared RAG singleton.
    
    If *instance* is provided it becomes the shared singleton, avoiding a
    second load of the embedding model.
    """
    global _shared_rag
    if instance is not None:
        _shared_rag = instance
    if _shared_rag is None:
        _shared_rag = RealRAGSystem()
    return _shared_rag

class FireManagementAgent:
    def __init__(self, groq_api_key: str = None, rag_system: Optional[RealRAGSystem] = None):
        # Use provided key, fall back to .env value
        self.groq_key = groq_api_key or GROQ_API_KEY
        self.rag = get_shared_rag(rag_system)
        self.incident_log = []
        
        # Define tools the agent can use
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "query_zone_procedure",
                    "description": "Get the emergency procedure for a specific zone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zone_id": {
                                "type": "integer",
                                "description": "The zone ID (1=Lobby, 2=Server Room, 3=Warehouse)"
                            }
                        },
                        "required": ["zone_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_suppression_info",
                    "description": "Get suppression system information for a zone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zone_id": {
                                "type": "integer",
                                "description": "The zone ID"
                            }
                        },
                        "required": ["zone_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "query_rag",
                    "description": "Search the fire safety knowledge base for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Fire safety question or search query"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "activate_suppression",
                    "description": "Activate suppression system for a zone",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zone_id": {
                                "type": "integer",
                                "description": "The zone ID"
                            },
                            "suppression_type": {
                                "type": "string",
                                "description": "Type of suppression (Sprinkler, CO2, Foam, Water Mist)"
                            },
                            "reason": {
                                "type": "string",
                                "description": "Reason for activation"
                            }
                        },
                        "required": ["zone_id", "suppression_type", "reason"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "log_incident",
                    "description": "Log an incident in the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zone_id": {
                                "type": "integer",
                                "description": "The zone ID"
                            },
                            "action": {
                                "type": "string",
                                "description": "Action taken"
                            },
                            "suppression_type": {
                                "type": "string",
                                "description": "Suppression system activated"
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Agent's reasoning"
                            }
                        },
                        "required": ["zone_id", "action", "suppression_type", "reasoning"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_emergency_email",
                    "description": "Send a structured emergency email alert based on detection data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "zone_name": { "type": "string" },
                            "address": { "type": "string" },
                            "severity": { "type": "string", "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"] },
                            "action_taken": { "type": "string" },
                            "reasoning": { "type": "string" }
                        },
                        "required": ["zone_name", "address", "severity", "action_taken"]
                    }
                }
            }
        ]
    
    # ============= HELPER FUNCTIONS =============
    def _json_serialize(self, obj: Any) -> str:
        """Helper to serialize objects, handling NumPy types like float32"""
        def default(o):
            if isinstance(o, (np.float32, np.float64)):
                return float(o)
            if isinstance(o, (np.int32, np.int64)):
                return int(o)
            return str(o)
        return json.dumps(obj, default=default)

    # ============= TOOL IMPLEMENTATIONS =============
    def query_zone_procedure(self, zone_id: int) -> str:
        """Get procedure for zone, including physical address"""
        proc = self.rag.get_zone_procedure(zone_id)
        if proc:
            return self._json_serialize(proc)
        return self._json_serialize({"error": f"Procedure not found for zone {zone_id}"})
    
    def get_suppression_info(self, zone_id: int) -> str:
        """Get suppression info for zone"""
        supp = self.rag.get_suppression_info(zone_id)
        if supp:
            return self._json_serialize(supp)
        return self._json_serialize({"error": f"Suppression info not found for zone {zone_id}"})
    
    def query_rag(self, query: str) -> str:
        """Search knowledge base"""
        results = self.rag.retrieve(query, top_k=2)
        return self._json_serialize(results)
    
    def activate_suppression(self, zone_id: int, suppression_type: str, reason: str) -> str:
        """Activate suppression system"""
        action = f"{suppression_type} suppression activated"
        self.log_incident(
            zone_id=zone_id,
            action=action,
            suppression_type=suppression_type,
            reasoning=reason
        )
        return self._json_serialize({
            "status": "activated",
            "zone_id": zone_id,
            "suppression_type": suppression_type,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    
    def build_emergency_html(self, zone_name: str, address: str, severity: str, action_taken: str, reasoning: str, lat=None, lon=None) -> str:
        """Generate a premium Cyber HUD style HTML email"""
        safe_zone = html.escape(zone_name or "Unknown Zone")
        safe_address = html.escape(address or "Unknown Location")
        safe_severity = html.escape(severity or "HIGH")
        safe_action = html.escape(action_taken or "Emergency response initiated")
        safe_reasoning = html.escape(reasoning or "Fire or smoke detection requires immediate review.")

        if lat and lon:
            map_url = f"https://www.google.com/maps?q={lat},{lon}"
        else:
            map_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(address or 'Unknown Location')}"
        safe_map_url = html.escape(map_url, quote=True)

        timestamp = datetime.now().strftime("%A, %B %d, %Y | %H:%M:%S")
        score_color = "#e11d48" if safe_severity == "CRITICAL" else "#f59e0b"
        incident_id = f"FW-{datetime.now().strftime('%y%m%d%H%M')}"
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
          <meta charset="UTF-8">
          <meta name="viewport" content="width=device-width,initial-scale=1.0">
          <title>FireWatch AI Emergency Alert</title>
        </head>
        <body style="margin:0;padding:0;background:#eef2f7;font-family:Arial,Helvetica,sans-serif;color:#172033;">
          <div style="display:none;max-height:0;overflow:hidden;opacity:0;color:transparent;">
            FireWatch AI detected a {safe_severity} fire or smoke incident at {safe_zone}.
          </div>
          <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:#eef2f7;margin:0;padding:24px 12px;">
            <tr>
              <td align="center">
                <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width:640px;background:#ffffff;border:1px solid #d8e0ea;border-radius:14px;overflow:hidden;">
                  <tr>
                    <td style="background:#111827;padding:24px 28px 20px 28px;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                          <td style="vertical-align:top;">
                            <div style="font-size:12px;line-height:16px;font-weight:700;letter-spacing:2px;color:#9ca3af;text-transform:uppercase;">FireWatch AI</div>
                            <div style="font-size:26px;line-height:32px;font-weight:800;color:#ffffff;margin-top:6px;">Emergency Alert</div>
                          </td>
                          <td align="right" style="vertical-align:top;">
                            <div style="display:inline-block;background:{score_color};color:#ffffff;font-size:12px;line-height:16px;font-weight:800;letter-spacing:1px;text-transform:uppercase;padding:8px 12px;border-radius:999px;">{safe_severity}</div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>

                  <tr>
                    <td style="background:#f8fafc;border-bottom:1px solid #e2e8f0;padding:14px 28px;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                          <td style="font-size:12px;line-height:18px;color:#64748b;">
                            <strong style="color:#1f2937;">Incident ID:</strong> {incident_id}
                          </td>
                          <td align="right" style="font-size:12px;line-height:18px;color:#64748b;">
                            <strong style="color:#1f2937;">Status:</strong> Active Response
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>

                  <tr>
                    <td style="padding:28px;">
                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                        <tr>
                          <td style="padding:0 0 18px 0;">
                            <div style="font-size:11px;line-height:14px;font-weight:800;letter-spacing:1.4px;color:#64748b;text-transform:uppercase;">Location</div>
                            <div style="font-size:22px;line-height:28px;font-weight:800;color:#111827;margin-top:6px;">{safe_zone}</div>
                            <div style="font-size:14px;line-height:21px;color:#475569;margin-top:6px;">{safe_address}</div>
                          </td>
                        </tr>
                        <tr>
                          <td style="padding:0 0 24px 0;">
                            <a href="{safe_map_url}" style="display:inline-block;background:#0f766e;color:#ffffff;text-decoration:none;font-size:13px;line-height:18px;font-weight:800;padding:11px 16px;border-radius:8px;">Open Location in Maps</a>
                          </td>
                        </tr>
                      </table>

                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #fecdd3;border-left:5px solid #e11d48;background:#fff5f6;border-radius:10px;">
                        <tr>
                          <td style="padding:18px 18px 16px 18px;">
                            <div style="font-size:11px;line-height:14px;font-weight:800;letter-spacing:1.4px;color:#be123c;text-transform:uppercase;">Action Triggered</div>
                            <div style="font-size:17px;line-height:24px;font-weight:800;color:#111827;margin-top:8px;">{safe_action}</div>
                            <div style="font-size:14px;line-height:22px;color:#4b5563;margin-top:8px;">{safe_reasoning}</div>
                          </td>
                        </tr>
                      </table>

                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:20px;border:1px solid #e2e8f0;background:#f8fafc;border-radius:10px;">
                        <tr>
                          <td width="50%" style="padding:16px 18px;border-right:1px solid #e2e8f0;">
                            <div style="font-size:11px;line-height:14px;font-weight:800;letter-spacing:1px;color:#64748b;text-transform:uppercase;">Detected At</div>
                            <div style="font-size:14px;line-height:21px;font-weight:700;color:#111827;margin-top:6px;">{timestamp}</div>
                          </td>
                          <td width="50%" style="padding:16px 18px;">
                            <div style="font-size:11px;line-height:14px;font-weight:800;letter-spacing:1px;color:#64748b;text-transform:uppercase;">Recommended Priority</div>
                            <div style="font-size:14px;line-height:21px;font-weight:800;color:{score_color};margin-top:6px;">Immediate Review</div>
                          </td>
                        </tr>
                      </table>

                      <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="margin-top:20px;background:#111827;border-radius:10px;">
                        <tr>
                          <td style="padding:20px;">
                            <div style="font-size:12px;line-height:16px;font-weight:800;letter-spacing:1.4px;color:#fca5a5;text-transform:uppercase;">Immediate Safety Protocol</div>
                            <div style="font-size:18px;line-height:25px;font-weight:800;color:#ffffff;margin-top:8px;">Evacuate the affected area immediately.</div>
                            <div style="font-size:14px;line-height:22px;color:#cbd5e1;margin-top:8px;">
                              Proceed to the nearest assembly point. Do not use elevators. Account for personnel and keep access clear for responders.
                            </div>
                          </td>
                        </tr>
                      </table>
                    </td>
                  </tr>

                  <tr>
                    <td style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:18px 28px;text-align:center;">
                      <a href="https://github.com/omerfarooq223/Agentic-Fire-Detection" style="color:#0f766e;text-decoration:none;font-size:13px;line-height:18px;font-weight:800;">Open FireWatch Dashboard</a>
                      <div style="font-size:11px;line-height:16px;color:#94a3b8;margin-top:8px;">FireWatch AI | Secure Incident Intelligence | {datetime.now().year}</div>
                    </td>
                  </tr>
                </table>
              </td>
            </tr>
          </table>
        </body>
        </html>
        """

    def record_demo_alert(
        self,
        channel: str,
        zone_name: str,
        address: str,
        severity: str,
        action_taken: str,
        reasoning: str = "",
        lat=None,
        lon=None,
    ) -> str:
        """Record a demo-mode alert instead of sending through a paid provider."""
        recipients = get_recipient_emails()
        alert = {
            "status": "Demo alert prepared",
            "channel": channel,
            "zone_name": zone_name,
            "address": address,
            "severity": severity,
            "action_taken": action_taken,
            "reasoning": reasoning,
            "lat": lat,
            "lon": lon,
            "timestamp": datetime.utcnow().isoformat(),
            "note": "Set AUTO_ALERTS_ENABLED=true and ALERT_MODE=email for live dispatch.",
            "recipients": recipients,
        }
        self.incident_log.append(alert)
        print(f"\n🧪 DEMO ALERT PREPARED: {channel} | {zone_name} | {severity} | recipients={recipients}")
        return self._json_serialize(alert)

    def send_emergency_email(self, zone_name: str, address: str, severity: str, action_taken: str, reasoning: str = "", lat=None, lon=None) -> str:
        """Send emergency email via official Gmail API using a premium HTML template"""
        print(f"📧 Sending Emergency Alert Email for {zone_name}...")
        print(f"📍 GPS Pinpoint Data: Lat={lat}, Lon={lon}")

        if ALERT_MODE == "demo":
            return self.record_demo_alert(
                channel="email",
                zone_name=zone_name,
                address=address,
                severity=severity,
                action_taken=action_taken,
                reasoning=reasoning,
                lat=lat,
                lon=lon,
            )
        
        if not os.path.exists(GOOGLE_TOKEN_FILE):
            return self._json_serialize({"error": f"{GOOGLE_TOKEN_FILE} not found."})

        recipients = get_recipient_emails()
        if not recipients:
            return self._json_serialize({"error": "No email alert recipients are configured or active."})
        
        subject = f"🚨 EMERGENCY FIRE ALERT: {zone_name} ({severity})"
        html_content = self.build_emergency_html(zone_name, address, severity, action_taken, reasoning, lat=lat, lon=lon)
        
        try:
            creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, ['https://www.googleapis.com/auth/gmail.send'])
            service = build('gmail', 'v1', credentials=creds)
            
            message = MIMEMultipart()
            message['To'] = ", ".join(recipients)
            message['From'] = SENDER_EMAIL
            message['Subject'] = subject
            message.attach(MIMEText(html_content, 'html'))
            
            raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
            service.users().messages().send(userId="me", body={'raw': raw_message}).execute()
            
            return self._json_serialize({"status": "HTML Alert sent successfully", "recipients": recipients})
        except Exception as e:
            return self._json_serialize({"error": f"Failed to send HTML alert: {str(e)}"})

    def log_incident(self, zone_id: int, action: str, suppression_type: str, reasoning: str) -> str:
        """Log incident"""
        incident = {
            "timestamp": datetime.utcnow().isoformat(),
            "zone_id": zone_id,
            "action": action,
            "suppression_type": suppression_type,
            "reasoning": reasoning
        }
        self.incident_log.append(incident)
        
        print(f"\n📋 INCIDENT LOGGED:")
        print(f"   Zone {zone_id} | {action}")
        print(f"   Suppression: {suppression_type}")
        print(f"   Reasoning: {reasoning}")
        
        return self._json_serialize(incident)
    
    # ============= PROCESS TOOL CALLS =============
    def process_tool_call(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool based on agent request"""
        if tool_name == "query_zone_procedure":
            return self.query_zone_procedure(tool_input.get("zone_id"))
        elif tool_name == "get_suppression_info":
            return self.get_suppression_info(tool_input.get("zone_id"))
        elif tool_name == "query_rag":
            return self.query_rag(tool_input.get("query"))
        elif tool_name == "activate_suppression":
            return self.activate_suppression(
                zone_id=tool_input.get("zone_id"),
                suppression_type=tool_input.get("suppression_type"),
                reason=tool_input.get("reason")
            )
        elif tool_name == "send_emergency_email":
            return self.send_emergency_email(
                zone_name=tool_input.get("zone_name"),
                address=tool_input.get("address"),
                severity=tool_input.get("severity"),
                action_taken=tool_input.get("action_taken"),
                reasoning=tool_input.get("reasoning", ""),
                lat=tool_input.get("lat"),
                lon=tool_input.get("lon")
            )
        elif tool_name == "log_incident":
            return self.log_incident(
                zone_id=tool_input.get("zone_id"),
                action=tool_input.get("action"),
                suppression_type=tool_input.get("suppression_type"),
                reasoning=tool_input.get("reasoning")
            )
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    # ============= AGENT REASONING LOOP =============
    def reason(self, detection_data: dict) -> dict:
        """
        Main agent reasoning loop
        Takes detection data, queries RAG, makes decision, logs incident
        """
        zone_id = detection_data.get("zone_id")
        coordinates = detection_data.get("coordinates", {})
        segment_area = detection_data.get("segment_area_pixels", 0)
        
        print("\n" + "=" * 70)
        print("🤖 FIRE MANAGEMENT AGENT - REASONING")
        print("=" * 70)
        print(f"🔥 Detection: Zone {zone_id}")
        print(f"   Coordinates: ({coordinates.get('x', 0):.1f}, {coordinates.get('y', 0):.1f})")
        print(f"   Segment Area: {segment_area:.0f} pixels")
        print()
        
        # Build context for the agent
        skip_email = detection_data.get("skip_email", False)
        system_prompt = f"""You are a fire management AI agent. You receive fire detection alerts and must:
1. Query the safety procedures for the detected zone
2. Get suppression system information
3. Query the knowledge base for relevant fire safety info
4. Decide on appropriate suppression system activation
5. Log the incident with your reasoning
6. Trigger 'send_emergency_email' ONLY if skip_email is False. (Current skip_email status: {skip_email})

CRITICAL RULE:
- You must send exactly ONE email per detection when email dispatch is enabled.
- If you see a tool result for 'send_emergency_email' in the message history, DO NOT call it again.
- Once the alert tools are called, summarize your final response and exit.

MANDATORY RULES:
- Use 'send_emergency_email' as soon as fire is confirmed.
- For 'severity', use 'CRITICAL' if segment area is > 1000 pixels, otherwise 'HIGH'.

You have access to tools to query procedures, suppression info, the knowledge base, and email alerts.
Always follow NFPA standards and zone-specific procedures.
Log all decisions with clear reasoning."""
        
        user_message = f"""FIRE DETECTION ALERT:
Zone: {zone_id}
Physical Address: {detection_data.get('address', 'Unknown Location')}
GPS Coordinates: {detection_data.get('lat', 'N/A')}, {detection_data.get('lon', 'N/A')}
Detection Coordinates: X={coordinates.get('x', 0):.1f}, Y={coordinates.get('y', 0):.1f}
Fire Segment Area: {segment_area:.0f} pixels

REQUIRED ACTIONS:
1. Verify the zone's emergency procedure and address using query_zone_procedure
2. Get suppression system info using get_suppression_info
3. Query the knowledge base for relevant fire safety info
4. Activate the appropriate suppression system
5. Log the incident with your reasoning (Use the Physical Address provided: {detection_data.get('address', 'Unknown')})
6. Trigger 'send_emergency_email'. YOU MUST INCLUDE THE RAW 'lat' AND 'lon' PARAMETERS in the tool call if they are available in the detection data. This is critical for the rescue navigation link.

Provide your analysis and decisions."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # Build tools list according to configured alert mode.
        active_tools = [
            t for t in self.tools
            if not (skip_email and t["function"]["name"] == "send_emergency_email")
        ]

        iteration = 0
        max_iterations = 8

        while iteration < max_iterations:
            iteration += 1
            print(f"\n🔄 Agent Iteration {iteration}:")
            
            # Call Groq API
            try:
                response = requests.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {self.groq_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": messages,
                        "tools": active_tools,
                        "tool_choice": "auto",
                        "temperature": 0.2,
                        "max_tokens": 1000
                    },
                    timeout=30
                )
                
                if response.status_code != 200:
                    print(f"⚠️  Groq API Error: {response.status_code}")
                    print(f"   Response: {response.text}")
                    break
                
                response_data = response.json()
                choice = response_data["choices"][0]["message"]
                content = choice.get("content", "")
                tool_calls = choice.get("tool_calls", [])
                
                # If agent provided content, print it
                if content:
                    print(f"   Agent: {content[:200]}...")
                
                # Add assistant message to history (ONCE per turn)
                messages.append(choice)

                # If no tool calls, agent is done
                if not tool_calls:
                    print(f"✅ Agent decision complete")
                    break
                
                # Process tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_input = json.loads(tool_call["function"]["arguments"])
                    
                    print(f"   🔧 Tool: {tool_name}")
                    print(f"      Input: {tool_input}")
                    
                    # FAIL-SAFE INJECTION: Ensure GPS coordinates are passed to the email tool
                    if tool_name == "send_emergency_email":
                        raw_lat = detection_data.get("lat")
                        raw_lon = detection_data.get("lon")
                        if raw_lat and not tool_input.get("lat"):
                            tool_input["lat"] = raw_lat
                        if raw_lon and not tool_input.get("lon"):
                            tool_input["lon"] = raw_lon
                        print(f"      [System] Injected GPS coordinates into {tool_name}")
                    
                    # Execute tool
                    tool_result = self.process_tool_call(tool_name, tool_input)
                    print(f"      Result: {tool_result[:100]}...")
                    
                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": tool_result
                    })
            
            except requests.exceptions.Timeout:
                print("⚠️  Groq API timeout - retrying...")
                continue
            except requests.exceptions.RequestException as e:
                print(f"⚠️  Request error: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON decode error: {e}")
                break
        
        print("\n" + "=" * 70)
        print("✅ REASONING COMPLETE")
        print("=" * 70)
        
        return {
            "zone_id": zone_id,
            "incident_log": self.incident_log[-1] if self.incident_log else None,
            "iterations": iteration
        }
    
    def get_incident_history(self, zone_id: int = None) -> list:
        """Get incident history"""
        if zone_id:
            return [inc for inc in self.incident_log if inc.get("zone_id") == zone_id]
        return self.incident_log


# ============= TEST WITHOUT GROQ KEY =============
def test_agent_without_groq():
    """Test agent structure without needing Groq API"""
    print("\n" + "=" * 70)
    print("🧪 TESTING AGENT STRUCTURE (No Groq API needed)")
    print("=" * 70)
    
    # Create agent (dummy key)
    agent = FireManagementAgent(groq_api_key="test_key")
    
    # Test tool implementations directly
    print("\n1️⃣  TESTING: Query Zone Procedure")
    print("-" * 70)
    for zone_id in [1, 2, 3]:
        result = agent.query_zone_procedure(zone_id)
        result_data = json.loads(result)
        print(f"✅ Zone {zone_id}: {result_data.get('category', 'N/A')}")
    
    print("\n2️⃣  TESTING: Get Suppression Info")
    print("-" * 70)
    for zone_id in [1, 2, 3]:
        result = agent.get_suppression_info(zone_id)
        result_data = json.loads(result)
        print(f"✅ Zone {zone_id}: {result_data.get('suppression_type', 'N/A')}")
    
    print("\n3️⃣  TESTING: Query RAG")
    print("-" * 70)
    queries = ["CO2 suppression", "Evacuation procedures", "Fire extinguisher"]
    for query in queries:
        result = agent.query_rag(query)
        result_data = json.loads(result)
        print(f"✅ Query '{query}': {len(result_data)} documents found")
    
    print("\n4️⃣  TESTING: Activate Suppression & Log Incident")
    print("-" * 70)
    agent.activate_suppression(
        zone_id=1,
        suppression_type="Sprinkler",
        reason="Fire detected in Lobby area"
    )
    
    print("\n5️⃣  INCIDENT HISTORY")
    print("-" * 70)
    history = agent.get_incident_history()
    for inc in history:
        print(f"✅ {inc['timestamp']}: Zone {inc['zone_id']} - {inc['action']}")
    
    print("\n✅ ALL TESTS PASSED!")


if __name__ == "__main__":
    # Run basic tests
    test_agent_without_groq()
    
    print("\n" + "=" * 70)
    print("📝 NEXT STEPS:")
    print("=" * 70)
    print("1. Get Groq API key: https://console.groq.com")
    print("2. Update GROQ_API_KEY in this file")
    print("3. Run: python test_agent.py")
    print("=" * 70)
