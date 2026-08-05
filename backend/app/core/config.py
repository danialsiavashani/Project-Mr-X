import os
from dotenv import load_dotenv

load_dotenv()

BEHAVIORPULSE_CLIENT_ID = os.getenv("BEHAVIORPULSE_CLIENT_ID")
BEHAVIORPULSE_API_KEY = os.getenv("BEHAVIORPULSE_API_KEY")
BEHAVIORPULSE_BASE_URL = os.getenv("BEHAVIORPULSE_BASE_URL")

if not BEHAVIORPULSE_CLIENT_ID or not BEHAVIORPULSE_API_KEY:
    print("WARNING: BehaviorPulse API credentials not set in .env")
if not BEHAVIORPULSE_BASE_URL:
    print("WARNING: BEHAVIORPULSE_BASE_URL not set in .env")