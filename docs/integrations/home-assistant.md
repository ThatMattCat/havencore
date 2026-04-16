# Home Assistant Integration

This guide covers integrating HavenCore with Home Assistant for comprehensive smart home control and automation.

## Overview

HavenCore provides deep integration with Home Assistant through:
- **Direct API Communication**: RESTful API calls to Home Assistant
- **Entity State Monitoring**: Real-time device status tracking
- **Service Execution**: Device control and automation triggers
- **Natural Language Interface**: Voice and text-based smart home control

## Prerequisites

### Home Assistant Setup
- **Home Assistant**: Version 2023.1+ (Core, Supervised, or OS)
- **Network Access**: HavenCore must be able to reach Home Assistant
- **Long-Lived Access Token**: For API authentication
- **SSL Certificate**: Recommended for secure communication

### HavenCore Configuration
- **Environment Variables**: Proper HA configuration in `.env`
- **Network Connectivity**: Accessible Home Assistant instance
- **Agent Service**: Running with Home Assistant tools enabled

## Initial Configuration

### 1. Generate Home Assistant Access Token

In Home Assistant:
1. Go to **Profile** → **Long-Lived Access Tokens**
2. Click **"Create Token"**
3. Give it a descriptive name: "HavenCore Integration"
4. Copy the generated token (you won't see it again)

### 2. Configure HavenCore Environment

Edit your `.env` file:
```bash
# Home Assistant Configuration
HAOS_URL="https://homeassistant.local:8123/api"  # Your HA URL
HAOS_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."  # Long-lived token

# Optional: For local/insecure connections
# HAOS_URL="http://192.168.1.100:8123/api"
```

**URL Format Examples**:
- Local: `http://192.168.1.100:8123/api`
- Domain: `https://homeassistant.local:8123/api`
- External: `https://your-domain.duckdns.org:8123/api`
- Nabu Casa: `https://abcd1234.ui.nabu.casa/api`

### 3. Test Connection

Verify the connection from HavenCore:
```bash
# Test from HavenCore agent container
docker compose exec agent curl -H "Authorization: Bearer YOUR_TOKEN" \
  "https://homeassistant.local:8123/api/"

# Should return: {"message": "API running."}
```

### 4. Restart HavenCore Services

```bash
# Restart to pick up new configuration
docker compose restart agent

# Verify agent health
curl http://localhost:6002/health
```

## Available Tools and Commands

HavenCore's HA MCP server (`mcp_homeassistant_tools`) currently exposes
**19 tools** covering generic REST/service calls, opinionated helpers for
common domains, WebSocket-powered registry + presence lookups, timer /
template / history / calendar access, and media-player transport control.
For a structured server-side reference (internals, config, troubleshooting),
see [MCP Home Assistant](../services/agent/tools/home-assistant.md). The section below focuses on
what the assistant can do for a user.

### Generic REST tools (always available)

| Tool | Purpose |
|------|---------|
| `ha_list_entities(domain?, area?, include_state?)` | Unified entity lookup. Filter by `domain` (all lights, all thermostats…), by `area` (everything in the kitchen), or both. Domain-only queries include state + curated attributes by default; area queries return bare entity_ids unless `include_state=true`. If a domain has no entities (e.g. `notify`, `tts`, `script`) the response includes a `hint` pointing at `ha_list_services` so the LLM doesn't stall on a dead end. |
| `ha_list_services(domain)` | Discover what services an HA integration exposes. Most common: `domain="notify"` to find `notify.mobile_app_*`. Also the right tool for `tts`, `script`, and other domains that publish actions rather than entities. |
| `ha_execute_service(entity_id, service, service_data?)` | Generic escape hatch — call any HA service on any entity. `service_data` is a JSON object (brightness, temperature, volume_level, etc.). |

**Voice examples**:

- "Show me all the lights" → `ha_list_entities` with `domain="light"`
- "What's in the kitchen?" → `ha_list_entities` with `area="kitchen"`
- "What notification services are set up?" → `ha_list_services` with `domain="notify"`
- "Start the vacuum cleaner" → `ha_execute_service` on a `vacuum.*` entity

**Common domains**: `light`, `switch`, `sensor`, `binary_sensor`, `climate`,
`cover`, `fan`, `lock`, `media_player`, `vacuum`, `camera`, `person`,
`device_tracker`, `timer`, `calendar`, `scene`, `script`, `automation`.

### Opinionated device / automation tools

These are higher-level wrappers around `ha_execute_service`. They exist so
the LLM passes structured arguments instead of free-form service data, and
so common patterns (multi-step climate control, light color+brightness)
are atomic tool calls.

| Tool | Purpose |
|------|---------|
| `ha_control_light(entity_id, state, brightness_pct?, color_name?, color_temp_kelvin?)` | Single call for on/off/toggle + brightness + color. Extras are ignored when turning off. |
| `ha_control_climate(entity_id, temperature?, hvac_mode?, fan_mode?)` | Thermostat control. Issues one HA service per non-null argument (`set_hvac_mode`, `set_temperature`, `set_fan_mode`) in that order. |
| `ha_activate_scene(scene_entity)` | `scene.turn_on` on the named scene. |
| `ha_trigger_script(script_entity, variables?)` | Runs a script, optionally passing `variables` as script-level variables. |
| `ha_trigger_automation(automation_entity)` | Manually fires an automation (same as pressing "Run" in the HA UI). |
| `ha_toggle_automation(entity_id, enabled)` | Enables or disables an automation — controls whether HA runs it on its triggers. |
| `ha_send_notification(service, message, title?, target?)` | Sends through a `notify.*` service (pass the service name without the `notify.` prefix). |

**Voice examples**:

- "Dim the living room to 30% warm white" → `ha_control_light` with
  `state=on`, `brightness_pct=30`, `color_temp_kelvin=2700`.
- "Set the bedroom to 68 and turn on heat" → `ha_control_climate` with
  `temperature=68`, `hvac_mode=heat`.
- "Activate movie night" → `ha_activate_scene(scene.movie_night)`.
- "Disable the porch light automation" → `ha_toggle_automation` with
  `enabled=false`.
- "Text Matt that dinner's ready" → `ha_send_notification` with the
  right `notify.mobile_app_*` service.

### Registry + presence tools (WebSocket-backed)

These use HA's WebSocket registry APIs to answer "what's in the kitchen?"
style questions and to summarize who's home.

| Tool | Purpose |
|------|---------|
| `ha_list_areas()` | Lists all HA areas (rooms / zones) with `area_id`, name, and aliases. |
| `ha_list_entities(area=…, domain?=…)` | Area-scoped listing (same tool as the generic entity lookup above, just with `area` set). Accepts area_id, name, or alias (case-insensitive). Entities inherit their **device's** area when they don't have a direct area of their own — this matters because most entities inherit rather than set area directly. Add `domain="light"` to narrow to one domain. |
| `ha_get_presence()` | Summary of all `person.*` and `device_tracker.*` entities with their current state (`home` / `not_home` / zone name). |

**Voice examples**:

- "What lights are in the kitchen?" → `ha_list_entities` with
  `area="kitchen"`, `domain="light"`.
- "Is anyone home?" → `ha_get_presence`.

### Timer / template / history / calendar tools

| Tool | Purpose |
|------|---------|
| `ha_set_timer(entity_id, duration?)` | Starts a `timer.*` helper. `duration` is `HH:MM:SS` (e.g. `"0:05:00"` for five minutes). Omit to use the timer's configured default. |
| `ha_cancel_timer(entity_id)` | Cancels a running timer. |
| `ha_evaluate_template(template)` | Renders a Jinja2 template server-side via `POST /api/template`. Use for compound checks (`"{{ is_state('binary_sensor.back_door','on') and is_state('person.matt','home') }}"`) or anything with `now()` / `states()`. |
| `ha_get_entity_history(entity_id, hours?)` | Recent state history (last N hours, capped at 168 = one week). Dense series are sampled down to ~200 points. |
| `ha_get_calendar_events(calendar_entity, days?)` | Upcoming events from a `calendar.*` entity (next N days, capped at 31). |

**Voice examples**:

- "Set a 10-minute kitchen timer" → `ha_set_timer(timer.kitchen, "0:10:00")`.
- "Has the garage door been open today?" → `ha_get_entity_history` with
  `entity_id="cover.garage"`, `hours=24`.
- "What's on the family calendar this week?" → `ha_get_calendar_events`
  with `calendar_entity="calendar.family"`, `days=7`.
- "Is the back door open and Matt home?" → `ha_evaluate_template` with
  a Jinja expression.

**Prerequisites**: `ha_set_timer` requires timer helpers defined in HA
(they're not dynamic — add them in `configuration.yaml` or the UI).
`ha_get_calendar_events` requires at least one calendar integration.

### Media player control

Home Assistant handles **transport control** (pause / resume / volume /
power / source) on any `media_player` entity via `ha_control_media_player`.

**Voice examples**:

- "Pause the TV"
- "Set volume to 50%"
- "Mute the living room speaker"
- "Turn off the bedroom TV"

**Supported actions**: `play`, `pause`, `stop`, `toggle`, `next`,
`previous`, `seek`, `shuffle`, `repeat`, `volume_set`, `volume_up`,
`volume_down`, `mute`, `unmute`, `turn_on`, `turn_off`, `select_source`.

Value units by action:

- `volume_set` → integer 0–100 (percent)
- `seek` → integer seconds from start
- `select_source` → source name string (e.g. `"HDMI 1"`)
- `shuffle` / `repeat` → boolean or `'off'` / `'all'` / `'one'`

**Library search and initiating playback of specific content is handled
by the Plex module, not Home Assistant.** See [Media Control](media-control.md)
for the Plex + HA split, required TV setup, and the optional wake/launch
mapping.

## Advanced Integration Patterns

### Chaining tools

The agent is free to chain tool calls across turns. Common patterns:

- **Area-aware control**: `ha_list_areas` → `ha_list_entities(area=…, domain="light")`
  → `ha_control_light` per entity. Lets the LLM turn off "all the lights
  in the bedroom" without pre-wired groups.
- **Notification discovery**: `ha_list_services(domain="notify")` →
  `ha_send_notification` against a discovered service name.
- **Camera → vision LLM**: `get_camera_snapshots` (from the MQTT module)
  returns URLs; the agent passes each URL to `query_multimodal_api`
  (from `mcp_general_tools`) to describe what the cameras see.

### Complex device control

- **Multi-room audio**: use `ha_list_entities(area=…, domain="media_player")`
  to enumerate `media_player` entities per room, then fan out
  `ha_control_media_player` calls.
- **Climate zones**: `ha_control_climate` operates on one entity at a
  time; "set all thermostats to 70" becomes one call per thermostat.
- **Security + locks**: use `ha_execute_service` against `lock.*` /
  `alarm_control_panel.*` entities for anything not covered by the
  opinionated helpers.

### Sensor and environmental data

- "What's the temperature in the living room?" → resolve the sensor via
  `ha_list_entities(area="living_room", domain="sensor", include_state=true)`,
  or read it directly with
  `ha_evaluate_template("{{ states('sensor.living_room_temperature') }}")`.
- "Has the garage been open today?" → `ha_get_entity_history` on the
  `cover.*` or `binary_sensor.*` entity over a 24-hour window.
- "Is anyone home?" → `ha_get_presence`.

## Natural Language Processing

### Entity Name Resolution
HavenCore automatically resolves natural language entity references:

- **Room-based**: "living room lights" → `light.living_room_*`
- **Friendly names**: "main thermostat" → `climate.main_thermostat`
- **Group references**: "all lights" → All entities in light domain
- **Partial matches**: "bedroom lamp" → `light.bedroom_lamp_1`

### Contextual Commands
Support for contextual and follow-up commands:

**Conversation Example**:
```
User: "Turn on the living room lights"
Assistant: "I've turned on the living room lights."
User: "Make them dimmer"
Assistant: "I've dimmed the living room lights to 50%."
User: "Change them to blue"
Assistant: "I've changed the living room lights to blue."
```

### Smart Defaults
Intelligent defaults based on context:

- **Time-based**: Evening commands may include dimming
- **Location-based**: "Turn on the lights" uses current room context
- **Activity-based**: "Movie mode" automatically adjusts multiple devices

## Troubleshooting

### Connection Issues

#### Cannot Connect to Home Assistant
```
Error: Failed to connect to Home Assistant API
```

**Solutions**:
1. **Check URL format**:
```bash
# Correct formats
HAOS_URL="https://homeassistant.local:8123/api"
HAOS_URL="http://192.168.1.100:8123/api"

# Incorrect (missing /api)
HAOS_URL="https://homeassistant.local:8123"
```

2. **Test connection manually**:
```bash
curl -I -H "Authorization: Bearer YOUR_TOKEN" "https://homeassistant.local:8123/api/"
```

3. **Check network connectivity**:
```bash
# From HavenCore container
docker compose exec agent ping homeassistant.local
docker compose exec agent curl -I https://homeassistant.local:8123
```

#### SSL Certificate Issues
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions**:
1. **Use proper SSL certificate** (recommended)
2. **Use HTTP instead of HTTPS** (local only):
```bash
HAOS_URL="http://192.168.1.100:8123/api"
```

3. **Add certificate to container** (advanced)

### Authentication Issues

#### Invalid Token Error
```
401 Unauthorized: Invalid or missing token
```

**Solutions**:
1. **Regenerate token** in Home Assistant
2. **Check token in environment**:
```bash
docker compose exec agent env | grep HAOS_TOKEN
```

3. **Verify token format** (should be long JWT-like string)

#### Token Expired
```
401 Unauthorized: Token has expired
```

**Solution**: Generate new long-lived access token in Home Assistant

### Service Execution Issues

#### Entity Not Found
```
FAILED: Light 'light.sitting_room_lamp' does not exist in Home Assistant.
Did you mean: light.sitting_room_floor_lamp, light.living_room_lamp?
No action was taken. Call ha_list_entities (with a `domain` or `area` filter)
to look up the correct entity_id, then retry. Do not guess entity names.
```

HA's service endpoint returns HTTP 200 with an empty body for *both*
non-existent entities and legitimate no-ops, so the HA MCP server
pre-flights every service call with a `GET /api/states/<entity_id>`. A 404
there becomes the `FAILED: ...` message above, and the service POST is
skipped entirely — nothing changes in HA.

When the assistant reports this, it means the entity ID it tried doesn't
exist. The pre-flight also does a fuzzy match against the live entity
registry in the same domain and surfaces up to a few near-matches in a
`Did you mean: …?` line — the LLM usually retries with one of those in
the next turn instead of stalling. If no good matches exist, the message
falls back to nudging the LLM toward `ha_list_entities`.

**If the entity *should* exist**:
1. **Check the exact ID** in Home Assistant → **Developer Tools → States**.
   IDs are case-sensitive and must match exactly
   (`light.kitchen_light_1`, not `light.kitchen`).
2. **Verify the domain prefix** (`light.`, `switch.`, `climate.`, etc.).
3. **Check the entity isn't disabled/hidden** in the entity registry —
   those are filtered out of area lookups.

#### Service Call Failed
```
Error: Service call failed
```

**Debug Steps**:
1. **Test in Home Assistant Developer Tools**:
   - Go to Developer Tools → Services
   - Try the same service call manually

2. **Check service parameters**:
   - Verify required parameters
   - Check parameter data types

3. **Review Home Assistant logs**:
   - Check Home Assistant logs for error details

### Performance Issues

#### Slow Response Times
```
Home Assistant commands taking too long
```

**Solutions**:
1. **Check Home Assistant performance**:
   - Monitor Home Assistant system resources
   - Check database size and optimization

2. **Network latency**:
   - Test network speed between HavenCore and HA
   - Consider local DNS resolution

3. **Optimize Home Assistant**:
   - Reduce sensor update frequencies
   - Clean up unused entities
   - Optimize automations

## Configuration Examples

### Complete `.env` Configuration
```bash
# Home Assistant Integration
HAOS_URL="https://homeassistant.local:8123/api"
HAOS_TOKEN="eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJmZjUzNmY4YjU5ZTQ0ZDczYWJjNjg5YzM5ZTM1MzNjNiIsImlhdCI6MTY5NzIxNDAwNCwiZXhwIjoyMDEyNTc0MDA0fQ.example_token_string"

# Optional: SSL verification (for self-signed certificates)
HAOS_VERIFY_SSL=true

# Optional: Connection timeouts (seconds)
HAOS_TIMEOUT=30

# Optional: Agent source IP (deprecated, for compatibility)
SOURCE_IP="10.0.0.100"
```

### Home Assistant Configuration
Add to your Home Assistant `configuration.yaml`:

```yaml
# Enable API (usually enabled by default)
api:

# HTTP configuration for external access
http:
  # Use SSL (recommended)
  ssl_certificate: /ssl/fullchain.pem
  ssl_key: /ssl/privkey.pem
  
  # Or allow internal access without SSL
  use_x_forwarded_for: true
  trusted_proxies:
    - 192.168.1.0/24  # Your local network
    - 172.16.0.0/12   # Docker networks

# Recorder configuration (for conversation history)
recorder:
  db_url: !secret postgres_url  # Optional: external database
  purge_keep_days: 30
  auto_purge: true
```

### Advanced Integration Script

Create a custom script for complex integrations:

```yaml
# Home Assistant: scripts.yaml
havencore_announce:
  alias: "HavenCore Announcement"
  sequence:
    - service: tts.speak
      data:
        entity_id: media_player.all_speakers
        message: "{{ message }}"
    - service: notify.mobile_app
      data:
        title: "HavenCore"
        message: "{{ message }}"

havencore_goodnight:
  alias: "HavenCore Good Night Routine"
  sequence:
    - service: light.turn_off
      data:
        entity_id: group.all_lights
    - service: climate.set_temperature
      data:
        entity_id: climate.main_thermostat
        temperature: 68
    - service: lock.lock
      data:
        entity_id: group.all_locks
    - service: alarm_control_panel.alarm_arm_night
      data:
        entity_id: alarm_control_panel.main
```

## Best Practices

### Security
1. **Use HTTPS** for Home Assistant access
2. **Rotate tokens** regularly
3. **Limit token scope** if possible
4. **Monitor access logs** in Home Assistant
5. **Use strong passwords** for Home Assistant accounts

### Performance
1. **Optimize entity names** for voice recognition
2. **Group related devices** for easier control
3. **Use scenes** for complex multi-device commands
4. **Cache frequently accessed** entity states
5. **Monitor Home Assistant resources**

### Reliability
1. **Test commands regularly** to ensure they work
2. **Handle device offline states** gracefully
3. **Implement retry logic** for failed commands
4. **Monitor Home Assistant availability**
5. **Have fallback methods** for critical controls

### User Experience
1. **Use natural entity names** ("Living Room Lamp" vs "light.living_room_lamp_1")
2. **Create intuitive scenes** ("Movie Night", "Bedtime")
3. **Group devices logically** by room or function
4. **Provide status feedback** for voice commands
5. **Test voice recognition** with actual users

---

**Next Steps**:
- [MCP Home Assistant](../services/agent/tools/home-assistant.md) - Server-side reference for the HA MCP module (tool internals, config, troubleshooting)
- [Media Control](media-control.md) - Plex + HA split for TV playback
- [Tool Development](../services/agent/tools/development.md) - Creating custom Home Assistant tools