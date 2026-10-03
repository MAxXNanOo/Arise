    const API_URL = "https://api.alertarise.online";

    // Map
    const map = L.map("map").setView([13.7563, 100.5018], 16);

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: "&copy; OpenStreetMap contributors",
        maxZoom: 19
    }).addTo(map);

    // Storage
    const incidentMarkers = {};
    let activeIncidents = {};

    // API Status
    async function checkAPI() {
        try {
            const response = await fetch(`${API_URL}/api/health`);

            if (!response.ok) {
                throw new Error("API offline");
            }

            document.getElementById("statusDot").className =
                "w-3 h-3 rounded-full bg-green-500";

            document.getElementById("statusText").innerText = "Online";
        } catch (error) {
            document.getElementById("statusDot").className =
                "w-3 h-3 rounded-full bg-red-500";

            document.getElementById("statusText").innerText = "Offline";
        }
    }

    // Load AI Data
    async function loadAIData() {
        try {
            const response = await fetch(`${API_URL}/api/ai`);

            if (!response.ok) return;

            const data = await response.json();
            updateDashboard(data);
        } catch (error) {
            console.error("AI data error:", error);
        }
    }

    // Update Dashboard
    function updateDashboard(data) {
        if (!data) return;

        const camera = data.camera;

        if (camera) {
            document.getElementById("cameraName").innerText =
                camera.name || "--";

            document.getElementById("cameraId").innerText =
                `Camera ID: ${camera.id}`;

            document.getElementById("location").innerText =
                camera.location || "--";

            document.getElementById("buildingFloor").innerText =
                `${camera.building || "--"} | ${camera.floor || "--"}`;
        }

        const actions = data.action_gru || [];

        document.getElementById("personCount").innerText = actions.length;

        updateObjects(data.object_detection || []);
        updateActions(actions);
        updateAudio(data.audio_yamnet);

        document.getElementById("rawJson").innerText =
            JSON.stringify(data, null, 2);

        document.getElementById("lastUpdate").innerText =
            data.timestamp || "--";
    }

    // Objects
    function updateObjects(objects) {
        const container = document.getElementById("objectList");

        if (!objects.length) {
            container.innerHTML = `
                <p class="text-slate-500">No objects</p>
            `;
            return;
        }

        container.innerHTML = objects.map(object => `
            <div class="flex justify-between bg-slate-800 rounded-lg p-3">
                <span>${object.class_name}</span>
                <span class="text-yellow-400">
                    ${(object.confidence * 100).toFixed(1)}%
                </span>
            </div>
        `).join("");
    }

    // Actions
    function updateActions(actions) {
        const container = document.getElementById("actionList");

        if (!actions.length) {
            container.innerHTML = `
                <p class="text-slate-500">No actions</p>
            `;
            return;
        }

        container.innerHTML = actions.map(action => `
            <div class="flex justify-between bg-slate-800 rounded-lg p-3">
                <span>
                    ID ${action.track_id} : ${action.class_name}
                </span>

                <span class="text-yellow-400">
                    ${(action.confidence * 100).toFixed(1)}%
                </span>
            </div>
        `).join("");
    }

    // Audio
    function updateAudio(audio) {
        if (!audio) return;

        const confidence = Number(audio.confidence || 0);

        document.getElementById("audioClass").innerText =
            audio.class_name || "No Audio";

        document.getElementById("audioConfidence").innerText =
            `${(confidence * 100).toFixed(1)}%`;

        document.getElementById("audioBar").style.width =
            `${confidence * 100}%`;
    }

    // Load Incidents
    async function loadIncidents() {
        try {
            const response = await fetch(`${API_URL}/api/alerts/active`);

            if (!response.ok) return;

            const data = await response.json();

            activeIncidents = {};

            data.incidents.forEach(incident => {
                activeIncidents[incident.incident_id] = incident;
                showIncident(incident);
            });

            Object.keys(incidentMarkers).forEach(id => {
                if (!activeIncidents[id]) {
                    map.removeLayer(incidentMarkers[id]);
                    delete incidentMarkers[id];
                }
            });

            updateIncidentList();
        } catch (error) {
            console.error("Incident error:", error);
        }
    }

    // Show Incident Marker
    function showIncident(incident) {
        const id = incident.incident_id;
        const camera = incident.camera;
        const evidence = incident.data.evidence;

        if (!camera) return;

        const lat = Number(camera.latitude);
        const lng = Number(camera.longitude);

        if (Number.isNaN(lat) || Number.isNaN(lng)) return;

        if (incidentMarkers[id]) return;

        const icon = L.divIcon({
            className: "incident-marker",
            html: `
                <div style="
                    width:34px;
                    height:34px;
                    border-radius:50%;
                    background:#dc2626;
                    border:4px solid white;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    color:white;
                    font-size:18px;
                    box-shadow:0 0 15px rgba(220,38,38,.8);
                ">
                    !
                </div>
            `,
            iconSize: [34, 34],
            iconAnchor: [17, 17]
        });

        const marker = L.marker([lat, lng], { icon }).addTo(map);
        const event = incident.incident || {};

        let imageHTML = "";

        if (incident.image) {
            imageHTML = `
                <img
                    src="data:image/jpeg;base64,${incident.image}"
                    class="w-full rounded-lg mt-3"
                >
            `;
        }

        let tempEvi = "";
        let temptempClass = [];
        evidence.forEach((evi, index) => {

            let check = true;
            temptempClass.forEach((data, index) => {
                if(data.class_name == evi.class_name){
                    check = false;
                }
            })
            if(check){
                tempEvi +=  `<p><b>เหตุการณ์ : </b> ${evi.class_name} ${(evi.confidence * 100).toFixed(1)}% </p>`
                temptempClass.push(evi);
            }

        });


        const popup = `
            <div style="width:300px;color:#111;">
                <h3 style="
                    font-size:20px;
                    font-weight:bold;
                    color:#dc2626;
                    margin-bottom:10px;
                ">
                    🚨 INCIDENT
                </h3>

                <p><b>Camera:</b> ${camera.name || camera.id}</p>
                <p><b>สถานที่:</b> ${camera.location || "--"}</p>
                <p><b>อาคาร:</b> ${camera.building || "--"}</p>
                <p><b>ชั้น:</b> ${camera.floor || "--"}</p>
                <p><b>รายละเอียด:</b> ${camera.description || "--"}</p>

                <hr style="margin:10px 0;">

                ${tempEvi}
                

                <p>
                    <b>Confidence:</b>
                    ${(Number(event.confidence || 0) * 100).toFixed(1)}%
                </p>


                <p><b>เวลา:</b> ${incident.timestamp || "--"}</p>

                ${imageHTML}

                <button
                    onclick="closeIncident('${id}')"
                    style="
                        margin-top:12px;
                        width:100%;
                        background:#dc2626;
                        color:white;
                        border:none;
                        padding:10px;
                        border-radius:8px;
                        cursor:pointer;
                        font-weight:bold;
                    "
                >
                    ปิดเหตุการณ์
                </button>
            </div>
        `;

    //         const popup = `
    //         <div style="width:300px;color:#111;">
    //             <h3 style="
    //                 font-size:20px;
    //                 font-weight:bold;
    //                 color:#dc2626;
    //                 margin-bottom:10px;
    //             ">
    //                 🚨 INCIDENT
    //             </h3>

    //             <p><b>Camera:</b> ${camera.name || camera.id}</p>
    //             <p><b>สถานที่:</b> ${camera.location || "--"}</p>
    //             <p><b>อาคาร:</b> ${camera.building || "--"}</p>
    //             <p><b>ชั้น:</b> ${camera.floor || "--"}</p>
    //             <p><b>รายละเอียด:</b> ${camera.description || "--"}</p>

    //             <hr style="margin:10px 0;">

                

    //             <p><b>เวลา:</b> ${incident.timestamp || "--"}</p>

    //             ${imageHTML}

    //             <button
    //                 onclick="closeIncident('${id}')"
    //                 style="
    //                     margin-top:12px;
    //                     width:100%;
    //                     background:#dc2626;
    //                     color:white;
    //                     border:none;
    //                     padding:10px;
    //                     border-radius:8px;
    //                     cursor:pointer;
    //                     font-weight:bold;
    //                 "
    //             >
    //                 ปิดเหตุการณ์
    //             </button>
    //         </div>
    //     `;

        marker.bindPopup(popup);

        incidentMarkers[id] = marker;

        map.setView([lat, lng], 18);
        marker.openPopup();
    }

    // Update Incident List
    function updateIncidentList() {
        const container = document.getElementById("incidentList");
        const incidents = Object.values(activeIncidents);

        document.getElementById("incidentCount").innerText =
            incidents.length;

        document.getElementById("incidentBadge").innerText =
            incidents.length;

        if (!incidents.length) {
            container.innerHTML = `
                <p class="text-slate-500 text-center py-10">
                    ไม่มีเหตุการณ์
                </p>
            `;
            return;
        }

        container.innerHTML = incidents.map(incident => {
            const camera = incident.camera || {};
            const event = incident.incident || {};

            return `
                <div
                    class="
                        bg-slate-800
                        rounded-xl
                        p-4
                        border
                        border-red-900
                        cursor-pointer
                        hover:bg-slate-700
                    "
                    onclick="focusIncident('${incident.incident_id}')"
                >
                    <div class="flex justify-between items-start">
                        <div>
                            <p class="font-bold text-red-400">
                                🚨 ${event.class_name || "Incident"}
                            </p>

                            <p class="text-sm text-slate-300 mt-1">
                                ${camera.name || camera.id}
                            </p>
                        </div>

                        <span class="text-xs bg-red-600 px-2 py-1 rounded">
                            ACTIVE
                        </span>
                    </div>

                    <p class="text-sm text-slate-400 mt-2">
                        ${camera.location || "--"}
                    </p>

                    <p class="text-xs text-slate-500 mt-1">
                        ${camera.building || "--"} |
                        ${camera.floor || "--"}
                    </p>

                    <p class="text-xs text-slate-500 mt-2">
                        ${incident.timestamp || "--"}
                    </p>
                </div>
            `;
        }).join("");
    }

    // Focus Incident
    function focusIncident(id) {
        const incident = activeIncidents[id];

        if (!incident) return;

        const camera = incident.camera;

        if (!camera || !incidentMarkers[id]) return;

        map.setView([
            Number(camera.latitude),
            Number(camera.longitude)
        ], 18);

        incidentMarkers[id].openPopup();
    }

    // Close Incident
    async function closeIncident(incidentId) {
        const confirmed = confirm("ยืนยันว่าต้องการปิดเหตุการณ์นี้?");

        if (!confirmed) return;

        try {
            const response = await fetch(
                `${API_URL}/api/alerts/${incidentId}/close`,
                {
                    method: "PATCH"
                }
            );

            const result = await response.json();

            if (!result.success) {
                alert("ไม่สามารถปิดเหตุการณ์ได้");
                return;
            }

            if (incidentMarkers[incidentId]) {
                map.removeLayer(incidentMarkers[incidentId]);
                delete incidentMarkers[incidentId];
            }

            delete activeIncidents[incidentId];

            updateIncidentList();
        } catch (error) {
            console.error("Close incident error:", error);
            alert("เชื่อมต่อ API ไม่สำเร็จ");
        }
    }

    // Update All
    async function updateAll() {
        await checkAPI();
        await loadAIData();
        await loadIncidents();
    }

    // Initial
    updateAll();

    // Realtime
    setInterval(loadAIData, 1000);
    setInterval(loadIncidents, 1000);
    setInterval(checkAPI, 3000);