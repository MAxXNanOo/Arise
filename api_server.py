from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import Dict
import threading


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="ARISE AI API",
    description="Realtime AI Monitoring API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# REALTIME AI DATA
# ============================================================

latest_data = {
    "timestamp": None,
    "camera": None,
    "object_detection": [],
    "action_gru": [],
    "audio_yamnet": {
        "class_name":
            "No Audio",
        "confidence":
            0.0
    }
}


# ============================================================
# INCIDENT STORAGE
# ============================================================

# เก็บ Incident ทั้งหมด
incidents: Dict[str, dict] = {}
incident_lock = threading.Lock()


# ============================================================
# HEALTH
# ============================================================

@app.get("/")
def root():

    return {
        "service" : "ARISE AI API" ,
        "status" : "online" ,
        "version" : "1.0.0"
    }


@app.get("/api/health")
def health():

    return {

        "status":
            "online",

        "service":
            "ARISE AI",

        "timestamp":
            datetime.now().isoformat()
    }


# ============================================================
# REALTIME AI DATA
# ============================================================

@app.get("/api/ai")
def get_ai_data():

    return latest_data


@app.post("/api/ai")
def update_ai_data(
    data: dict
):

    global latest_data

    latest_data = data

    return {

        "success":
            True,

        "timestamp":
            datetime.now().isoformat()
    }


# ============================================================
# CREATE INCIDENT
# ============================================================

@app.post("/api/alerts")
def create_alert(
    data: dict
):

    incident_id = data.get(
        "incident_id"
    )

    if not incident_id:

        return {

            "success":
                False,

            "error":
                "missing incident_id"
        }


    # บังคับ status เป็น active
    data["status"] = "active"


    # เวลาที่ server รับ
    data["received_at"] = \
        datetime.now().isoformat()


    with incident_lock:

        incidents[
            incident_id
        ] = data


    camera = data.get(
        "camera",
        {}
    )

    incident = data.get(
        "incident",
        {}
    )


    print()
    print("=" * 75)
    print("🚨 NEW INCIDENT")
    print("=" * 75)

    print(
        f"Incident ID : "
        f"{incident_id}"
    )

    print(
        f"Type        : "
        f"{incident.get('type')}"
    )

    print(
        f"Class       : "
        f"{incident.get('class_name')}"
    )

    print(
        f"Confidence  : "
        f"{incident.get('confidence')}"
    )

    print(
        f"Camera      : "
        f"{camera.get('id')}"
    )

    print(
        f"Location    : "
        f"{camera.get('location')}"
    )

    print(
        f"Building    : "
        f"{camera.get('building')}"
    )

    print(
        f"Floor       : "
        f"{camera.get('floor')}"
    )

    print(
        f"Latitude    : "
        f"{camera.get('latitude')}"
    )

    print(
        f"Longitude   : "
        f"{camera.get('longitude')}"
    )

    print("=" * 75)


    return {

        "success":
            True,

        "incident_id":
            incident_id,

        "status":
            "active"
    }


# ============================================================
# GET ALL INCIDENTS
# ============================================================

@app.get("/api/alerts")
def get_alerts():

    with incident_lock:

        all_incidents = list(
            incidents.values()
        )

    return {

        "count":
            len(all_incidents),

        "incidents":
            all_incidents
    }


# ============================================================
# GET ACTIVE INCIDENTS
# ============================================================

@app.get("/api/alerts/active")
def get_active_alerts():

    with incident_lock:

        active = [

            incident

            for incident
            in incidents.values()

            if incident.get(
                "status"
            ) == "active"
        ]

    return {

        "count":
            len(active),

        "incidents":
            active
    }


# ============================================================
# GET SINGLE INCIDENT
# ============================================================

@app.get(
    "/api/alerts/{incident_id}"
)
def get_alert(
    incident_id: str
):

    with incident_lock:

        incident = incidents.get(
            incident_id
        )

    if incident is None:

        return {

            "success":
                False,

            "error":
                "incident not found"
        }


    return {

        "success":
            True,

        "incident":
            incident
    }


# ============================================================
# CLOSE INCIDENT
# ============================================================

@app.patch(
    "/api/alerts/{incident_id}/close"
)
def close_alert(
    incident_id: str
):

    with incident_lock:

        if incident_id not in incidents:

            return {

                "success":
                    False,

                "error":
                    "incident not found"
            }


        incidents[
            incident_id
        ]["status"] = "closed"


        incidents[
            incident_id
        ]["closed_at"] = \
            datetime.now().isoformat()


        incident = incidents[
            incident_id
        ]


    print()
    print(
        f"✅ INCIDENT CLOSED: "
        f"{incident_id}"
    )


    return {

        "success":
            True,

        "incident_id":
            incident_id,

        "status":
            "closed",

        "closed_at":
            incident["closed_at"]
    }


# ============================================================
# DELETE INCIDENT
# ============================================================

@app.delete(
    "/api/alerts/{incident_id}"
)
def delete_alert(
    incident_id: str
):

    with incident_lock:

        if incident_id not in incidents:

            return {

                "success":
                    False,

                "error":
                    "incident not found"
            }


        del incidents[
            incident_id
        ]


    return {

        "success":
            True,

        "incident_id":
            incident_id
    }


# ============================================================
# SERVER START
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "api_server:app",

        host="0.0.0.0",

        port=5500,

        reload=False
    )
