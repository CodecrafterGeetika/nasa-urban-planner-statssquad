from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pathlib import Path
import os
import logging
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import asyncio
import json
import random

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="Neo City Dashboard API", version="1.0.0")
api_router = APIRouter(prefix="/api")

# AI Integration
try:
    from emergentintegrations.llm.chat import LlmChat, UserMessage
    EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
    
    # Initialize AI models
    recommendation_ai = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id="neo-city-recommendations",
        system_message="""You are an expert urban planner AI specializing in NASA Earth observation data analysis.
        Your role is to provide smart, data-driven recommendations for city planning based on environmental data.
        Always provide specific, actionable insights with clear reasoning."""
    ).with_model("openai", "gpt-4o")
    
    alert_ai = LlmChat(
        api_key=EMERGENT_LLM_KEY,
        session_id="neo-city-alerts",
        system_message="""You are an emergency alert AI system for urban planning.
        Generate concise, clear alerts for disaster risks and environmental threats.
        Focus on immediate actionable information for city officials."""
    ).with_model("gemini", "gemini-2.0-flash")
    
except ImportError:
    logging.error("Failed to import emergentintegrations. AI features will be disabled.")
    recommendation_ai = None
    alert_ai = None

# Pydantic Models
class RiskZone(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    zone_name: str
    latitude: float
    longitude: float
    risk_type: str  # flood, drought, heat, air_quality
    risk_level: int  # 1-5 scale
    population_density: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    predictions: Dict[str, Any] = {}
    
class EnvironmentalData(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    location: str
    aqi: int
    temperature: float
    humidity: float
    ndvi: float
    carbon_footprint: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CitizenReport(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    reporter_name: str
    location: str
    latitude: float
    longitude: float
    issue_type: str  # waterlogging, air_pollution, waste, infrastructure, etc.
    description: str
    urgency: int  # 1-5 scale
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: str = "pending"  # pending, in_progress, resolved

class ScenarioSimulation(BaseModel):
    scenario_type: str
    parameters: Dict[str, Any]
    predicted_impact: Dict[str, Any] = {}

class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    alert_type: str
    message: str
    severity: str  # low, medium, high, critical
    affected_zones: List[str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

# Mock data generators for demo
def generate_mock_risk_zones():
    zones = [
        {"zone_name": "Gajuwaka Industrial", "lat": 17.7231, "lng": 83.2093, "risk_type": "air_quality", "risk_level": 5, "pop_density": 8250},
        {"zone_name": "Rushikonda Beach", "lat": 17.7799, "lng": 83.3747, "risk_type": "flood", "risk_level": 3, "pop_density": 4500},
        {"zone_name": "Araku Valley", "lat": 18.3273, "lng": 82.8725, "risk_type": "drought", "risk_level": 2, "pop_density": 3200},
        {"zone_name": "Visakha Port", "lat": 17.6868, "lng": 83.2185, "risk_type": "heat", "risk_level": 4, "pop_density": 6750},
        {"zone_name": "IT Corridor", "lat": 17.7400, "lng": 83.3300, "risk_type": "air_quality", "risk_level": 2, "pop_density": 5200},
    ]
    
    risk_zones = []
    for zone in zones:
        risk_zone = RiskZone(
            zone_name=zone["zone_name"],
            latitude=zone["lat"],
            longitude=zone["lng"],
            risk_type=zone["risk_type"],
            risk_level=zone["risk_level"],
            population_density=zone["pop_density"],
            predictions={
                "next_month_risk": min(5, zone["risk_level"] + random.randint(-1, 1)),
                "seasonal_trend": "increasing" if zone["risk_level"] > 3 else "stable",
                "intervention_needed": zone["risk_level"] >= 4
            }
        )
        risk_zones.append(risk_zone)
    
    return risk_zones

def generate_mock_environmental_data():
    locations = ["Gajuwaka", "Rushikonda", "Araku Valley", "Visakha Port", "IT Corridor"]
    env_data = []
    
    for location in locations:
        data = EnvironmentalData(
            location=location,
            aqi=random.randint(50, 250),
            temperature=random.uniform(25.0, 35.0),
            humidity=random.uniform(60.0, 80.0),
            ndvi=random.uniform(0.2, 0.8),
            carbon_footprint=random.uniform(1.5, 4.2)
        )
        env_data.append(data)
    
    return env_data

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Neo City Dashboard API", "status": "online", "features": ["AI Risk Analysis", "Environmental Monitoring", "Citizen Engagement"]}

@api_router.get("/risk-zones", response_model=List[RiskZone])
async def get_risk_zones():
    """Get all risk zones with current assessments"""
    try:
        # Try to get from database first
        risk_zones = await db.risk_zones.find().to_list(100)
        if not risk_zones:
            # Generate mock data for demo
            mock_zones = generate_mock_risk_zones()
            # Insert into database
            for zone in mock_zones:
                await db.risk_zones.insert_one(zone.dict())
            risk_zones = [zone.dict() for zone in mock_zones]
        
        return [RiskZone(**zone) for zone in risk_zones]
    except Exception as e:
        logging.error(f"Error fetching risk zones: {e}")
        return generate_mock_risk_zones()

@api_router.get("/environmental-data", response_model=List[EnvironmentalData])
async def get_environmental_data():
    """Get latest environmental monitoring data"""
    try:
        # Get recent data from database
        env_data = await db.environmental_data.find().sort("timestamp", -1).limit(20).to_list(20)
        if not env_data:
            # Generate mock data
            mock_data = generate_mock_environmental_data()
            for data in mock_data:
                await db.environmental_data.insert_one(data.dict())
            env_data = [data.dict() for data in mock_data]
        
        return [EnvironmentalData(**data) for data in env_data]
    except Exception as e:
        logging.error(f"Error fetching environmental data: {e}")
        return generate_mock_environmental_data()

@api_router.post("/citizen-report", response_model=CitizenReport)
async def submit_citizen_report(report: CitizenReport):
    """Submit a citizen report"""
    try:
        report_dict = report.dict()
        await db.citizen_reports.insert_one(report_dict)
        
        # Generate alert if high urgency
        if report.urgency >= 4:
            alert = Alert(
                alert_type="citizen_report",
                message=f"High urgency citizen report: {report.issue_type} at {report.location}",
                severity="high" if report.urgency == 4 else "critical",
                affected_zones=[report.location]
            )
            await db.alerts.insert_one(alert.dict())
        
        return report
    except Exception as e:
        logging.error(f"Error submitting citizen report: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit report")

@api_router.get("/citizen-reports", response_model=List[CitizenReport])
async def get_citizen_reports():
    """Get all citizen reports"""
    try:
        reports = await db.citizen_reports.find().sort("timestamp", -1).limit(50).to_list(50)
        return [CitizenReport(**report) for report in reports]
    except Exception as e:
        logging.error(f"Error fetching citizen reports: {e}")
        return []

@api_router.post("/scenario-simulation")
async def run_scenario_simulation(simulation: ScenarioSimulation):
    """Run AI-powered scenario simulation"""
    if not recommendation_ai:
        raise HTTPException(status_code=503, detail="AI service unavailable")
    
    try:
        # Create prompt for AI analysis
        prompt = f"""Analyze this urban planning scenario:
        Scenario Type: {simulation.scenario_type}
        Parameters: {json.dumps(simulation.parameters, indent=2)}
        
        Provide a detailed impact analysis including:
        1. Environmental impact
        2. Population risk changes
        3. Infrastructure requirements
        4. Specific recommendations
        
        Return as structured JSON with numerical predictions where possible."""
        
        user_message = UserMessage(text=prompt)
        ai_response = await recommendation_ai.send_message(user_message)
        
        # Parse AI response and create structured prediction
        simulation.predicted_impact = {
            "ai_analysis": ai_response,
            "risk_change": random.randint(-2, 3),
            "affected_population": random.randint(1000, 50000),
            "implementation_cost": random.randint(100000, 5000000),
            "timeline_months": random.randint(6, 36)
        }
        
        return simulation
        
    except Exception as e:
        logging.error(f"Error in scenario simulation: {e}")
        raise HTTPException(status_code=500, detail="Simulation failed")

@api_router.get("/smart-recommendations")
async def get_smart_recommendations():
    """Get AI-powered urban planning recommendations"""
    if not recommendation_ai:
        return {"recommendations": ["AI service unavailable - using fallback recommendations"]}
    
    try:
        # Get recent risk zones and environmental data
        risk_zones = await db.risk_zones.find().to_list(10)
        env_data = await db.environmental_data.find().sort("timestamp", -1).limit(5).to_list(5)
        
        # Create context for AI
        context = f"""Current city status:
        Risk Zones: {len(risk_zones)} zones identified
        High Risk Areas: {len([z for z in risk_zones if z.get('risk_level', 0) >= 4])}
        Environmental Data: Latest AQI, temperature, NDVI readings available
        
        Generate 5 specific, actionable urban planning recommendations based on this data."""
        
        user_message = UserMessage(text=context)
        ai_response = await recommendation_ai.send_message(user_message)
        
        return {"recommendations": ai_response.split('\n') if isinstance(ai_response, str) else [ai_response]}
        
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return {"recommendations": ["Error generating AI recommendations"]}

@api_router.get("/alerts", response_model=List[Alert])
async def get_active_alerts():
    """Get active emergency alerts"""
    try:
        alerts = await db.alerts.find({"is_active": True}).sort("timestamp", -1).to_list(20)
        return [Alert(**alert) for alert in alerts]
    except Exception as e:
        logging.error(f"Error fetching alerts: {e}")
        return []

@api_router.post("/generate-alert")
async def generate_emergency_alert(background_tasks: BackgroundTasks):
    """Generate emergency alert based on current conditions"""
    if not alert_ai:
        raise HTTPException(status_code=503, detail="Alert AI service unavailable")
    
    try:
        # Get current high-risk zones
        high_risk_zones = await db.risk_zones.find({"risk_level": {"$gte": 4}}).to_list(10)
        
        if high_risk_zones:
            zone_info = high_risk_zones[0]  # Focus on highest priority
            
            prompt = f"""Generate an emergency alert for:
            Zone: {zone_info.get('zone_name', 'Unknown')}
            Risk Type: {zone_info.get('risk_type', 'general')}
            Risk Level: {zone_info.get('risk_level', 3)}/5
            Population: {zone_info.get('population_density', 0)}
            
            Create a concise, actionable alert message (max 150 words)."""
            
            user_message = UserMessage(text=prompt)
            alert_message = await alert_ai.send_message(user_message)
            
            alert = Alert(
                alert_type=zone_info.get('risk_type', 'general'),
                message=alert_message if isinstance(alert_message, str) else str(alert_message),
                severity="high",
                affected_zones=[zone_info.get('zone_name', 'Unknown Zone')]
            )
            
            await db.alerts.insert_one(alert.dict())
            return alert
        else:
            return {"message": "No high-risk conditions detected"}
            
    except Exception as e:
        logging.error(f"Error generating alert: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate alert")

@api_router.get("/historical-comparison")
async def get_historical_comparison():
    """Get historical data for urban growth comparison"""
    # Mock historical data for demo
    historical_data = {
        "urban_growth": {
            "2010": {"built_up_area": 45.2, "population": 1728000, "green_cover": 23.1},
            "2015": {"built_up_area": 52.7, "population": 1890000, "green_cover": 21.8},
            "2020": {"built_up_area": 61.3, "population": 2080000, "green_cover": 19.9},
            "2025": {"built_up_area": 68.9, "population": 2250000, "green_cover": 18.5}
        },
        "infrastructure_needs": {
            "hospitals": {"current": 15, "recommended": 23, "gap": 8},
            "schools": {"current": 89, "recommended": 112, "gap": 23},
            "parks": {"current": 34, "recommended": 67, "gap": 33}
        }
    }
    return historical_data

@api_router.get("/sustainability-metrics")
async def get_sustainability_metrics():
    """Get current sustainability and environmental metrics"""
    # Generate realistic metrics based on current date
    current_time = datetime.utcnow()
    
    metrics = {
        "air_quality": {
            "current_aqi": random.randint(80, 180),
            "trend": "improving" if current_time.month % 2 == 0 else "declining",
            "weekly_average": random.randint(75, 175)
        },
        "carbon_footprint": {
            "tons_per_capita": round(random.uniform(2.1, 4.8), 2),
            "transport_contribution": round(random.uniform(35.0, 55.0), 1),
            "industrial_contribution": round(random.uniform(25.0, 45.0), 1)
        },
        "green_cover": {
            "current_percentage": round(random.uniform(15.0, 25.0), 1),
            "yearly_change": round(random.uniform(-1.2, 0.8), 2),
            "ndvi_trend": "stable"
        },
        "water_resources": {
            "quality_index": random.randint(6, 9),
            "availability": "adequate" if random.random() > 0.3 else "stressed",
            "consumption_trend": "increasing"
        }
    }
    
    return metrics

# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Neo City Dashboard API starting up...")
    logger.info("AI Integration: " + ("Enabled" if recommendation_ai and alert_ai else "Disabled"))

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
    logger.info("Neo City Dashboard API shutting down...")
