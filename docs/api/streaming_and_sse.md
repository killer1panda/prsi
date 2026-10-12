# Live Streaming & Server-Sent Events (SSE)

## Overview
PRSI provides real-time event streaming via **Server-Sent Events (SSE)** at `GET /events`. This standard HTTP-based streaming interface allows web and desktop dashboards to consume live threat scores without the overhead and state management complexity of WebSockets.

---

## 📡 Endpoint Details

- **Path:** `/events`
- **Method:** `GET`
- **Content-Type:** `text/event-stream`
- **Headers:**
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`
  - `X-Accel-Buffering: no` (disables Nginx/reverse proxy response buffering)

---

## 💻 Client Integration Examples

### 1. Browser / React Native (`EventSource`)
```typescript
const eventSource = new EventSource("http://localhost:8000/events");

eventSource.onopen = () => {
  console.log("Connected to PRSI Live Threat Feed");
};

eventSource.onmessage = (event) => {
  try {
    const payload = JSON.parse(event.data);
    console.log(`[Event ${payload.id}] Score: ${payload.doom_score} | ${payload.text}`);
    // Update live UI state
  } catch (err) {
    console.error("Failed to parse event JSON:", err);
  }
};

eventSource.onerror = (err) => {
  console.warn("SSE connection interrupted. Reconnecting...", err);
};
```

### 2. Python (`httpx` Streaming)
```python
import httpx
import json

async def stream_live_events():
    url = "http://localhost:8000/events"
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("GET", url) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    event_data = json.loads(data_str)
                    print(f"[{event_data['risk_level']}] {event_data['text']}")
```

---

## 🔄 Reconnection & Resiliency Protocol
1. **Heartbeat:** Events are emitted at an interval of 4.0 seconds. If no social media events occur, synthetic calibration posts are generated to keep the connection alive.
2. **Auto-Reconnect:** Web clients retry connection automatically with exponential backoff if the gateway drops the connection.
