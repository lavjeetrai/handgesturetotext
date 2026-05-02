const video = document.querySelector("#video");
const canvas = document.querySelector("#captureCanvas");
const statusEl = document.querySelector("#status");
const symbolEl = document.querySelector("#symbol");
const sentenceEl = document.querySelector("#sentence");
const skeletonEl = document.querySelector("#skeleton");
const startBtn = document.querySelector("#startBtn");
const stopBtn = document.querySelector("#stopBtn");
const speakBtn = document.querySelector("#speakBtn");
const clearBtn = document.querySelector("#clearBtn");
const suggestionBtns = [...document.querySelectorAll(".suggestion")];

const sessionIdKey = "sign-webapp-session-id";
let sessionId = localStorage.getItem(sessionIdKey);
if (!sessionId) {
  sessionId = crypto.randomUUID();
  localStorage.setItem(sessionIdKey, sessionId);
}

let stream = null;
let timer = null;
let inFlight = false;

function setStatus(message) {
  statusEl.textContent = message;
}

function updateView(data) {
  if (data.symbol) {
    symbolEl.textContent = data.symbol.trim() || "-";
  }

  sentenceEl.value = data.sentence || "";

  if (data.skeleton) {
    skeletonEl.src = data.skeleton;
  }

  suggestionBtns.forEach((button, index) => {
    const text = (data.suggestions && data.suggestions[index]) || "";
    button.textContent = text;
    button.disabled = !text;
  });

  if (data.status === "no_hand") {
    setStatus("Show one hand clearly in the camera");
  } else if (data.status === "ok") {
    setStatus("Recognizing");
  } else if (data.status === "error") {
    setStatus(data.message || "Prediction error");
  }
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.message || "Request failed");
  }
  return data;
}

async function predictOnce() {
  if (!stream || inFlight || video.readyState < 2) {
    return;
  }

  inFlight = true;
  try {
    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth || 480;
    canvas.height = video.videoHeight || 360;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL("image/jpeg", 0.78);
    const data = await postJson("/api/predict", { session_id: sessionId, image });
    updateView(data);
  } catch (error) {
    setStatus(error.message);
  } finally {
    inFlight = false;
  }
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("Camera starting");
  timer = window.setInterval(predictOnce, 420);
}

function stopCamera() {
  if (timer) {
    window.clearInterval(timer);
    timer = null;
  }
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Camera stopped");
}

startBtn.addEventListener("click", async () => {
  try {
    await startCamera();
  } catch (error) {
    setStatus(error.message);
  }
});

stopBtn.addEventListener("click", stopCamera);

clearBtn.addEventListener("click", async () => {
  const data = await postJson("/api/reset", { session_id: sessionId });
  updateView(data);
  setStatus("Cleared");
});

speakBtn.addEventListener("click", () => {
  const text = sentenceEl.value.trim();
  if (!text || !window.speechSynthesis) {
    return;
  }
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
});

suggestionBtns.forEach((button) => {
  button.addEventListener("click", async () => {
    const data = await postJson(`/api/suggest/${button.dataset.index}`, {
      session_id: sessionId,
    });
    updateView(data);
  });
});
