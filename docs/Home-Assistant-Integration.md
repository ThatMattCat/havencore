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

### Entity State Tools

#### Get Domain Entity States
Retrieve all entities for a specific domain (lights, switches, sensors, etc.):

**Voice Commands**:
- "Show me all the lights"
- "What sensors do we have?"
- "List all switches"

**API Usage**:
```bash
curl -X POST http://localhost/v1/chat/completions \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Show me all the lights in the house"}
    ]
  }'
```

**Supported Domains**:
- `light` - Light entities
- `switch` - Switch entities  
- `sensor` - Sensor entities
- `binary_sensor` - Binary sensors
- `climate` - Climate control
- `cover` - Covers/blinds
- `fan` - Fan entities
- `lock` - Lock entities
- `media_player` - Media devices
- `vacuum` - Vacuum cleaners
- `camera` - Camera entities

### Service Execution Tools

#### Execute Home Assistant Services
Control devices and trigger automations:

**Voice Commands**:
- "Turn on the living room lights"
- "Set the thermostat to 72 degrees"
- "Close the bedroom blinds"
- "Start the vacuum cleaner"

**Service Examples**:

##### Light Control
```json
{
  "domain": "light",
  "service": "turn_on",
  "service_data": {
    "entity_id": "light.living_room",
    "brightness": 255,
    "color_name": "blue"
  }
}
```

##### Climate Control
```json
{
  "domain": "climate", 
  "service": "set_temperature",
  "service_data": {
    "entity_id": "climate.main_thermostat",
    "temperature": 72
  }
}
```

##### Switch Control
```json
{
  "domain": "switch",
  "service": "toggle",
  "service_data": {
    "entity_id": "switch.coffee_maker"
  }
}
```

### Media Player Control

#### Basic Media Controls
Control media playback devices:

**Voice Commands**:
- "Play music in the living room"
- "Pause the TV"
- "Set volume to 50%"
- "Next song"

**Available Controls**:
- `play` - Start playback
- `pause` - Pause playback
- `stop` - Stop playback
- `previous_track` - Previous track
- `next_track` - Next track
- `volume_up` - Increase volume
- `volume_down` - Decrease volume
- `volume_set` - Set specific volume
- `mute` - Toggle mute

#### Media Player Status
Get current status of media devices:

**Voice Commands**:
- "What's playing in the living room?"
- "Show me all media players"
- "Is the TV on?"

**Status Information**:
- Current media title and artist
- Playback state (playing, paused, idle)
- Volume level and mute status
- Source input information

#### Play Media Content
Play specific content on media devices:

**Voice Commands**:
- "Play Spotify playlist 'Chill Music' in the living room"
- "Play the news on the kitchen speaker"
- "Start Netflix on the TV"

**Supported Content Types**:
- Music (Spotify, Apple Music, local files)
- Podcasts and radio stations
- Video content (Netflix, YouTube, Plex)
- Audio announcements and TTS

## Advanced Integration Patterns

### Scene and Automation Control

#### Scene Activation
Control Home Assistant scenes:

**Voice Commands**:
- "Activate movie night scene"
- "Turn on bedtime mode"
- "Set romantic lighting"

**Implementation**:
```json
{
  "domain": "scene",
  "service": "turn_on", 
  "service_data": {
    "entity_id": "scene.movie_night"
  }
}
```

#### Automation Triggers
Trigger Home Assistant automations:

**Voice Commands**:
- "Run the morning routine"
- "Activate security mode"
- "Start the bedtime sequence"

**Implementation**:
```json
{
  "domain": "automation",
  "service": "trigger",
  "service_data": {
    "entity_id": "automation.morning_routine"
  }
}
```

### Complex Device Control

#### Multi-Room Audio
Control audio across multiple rooms:

**Voice Commands**:
- "Play jazz music throughout the house"
- "Join the kitchen speaker to the living room"
- "Stop music in all bedrooms"

#### Climate Zones
Control multiple climate zones:

**Voice Commands**:
- "Set all thermostats to 70 degrees"
- "Turn on heat in the bedrooms"
- "What's the temperature upstairs?"

#### Security System Integration
Control security systems and locks:

**Voice Commands**:
- "Lock all the doors"
- "Arm the security system"
- "Show me the front door camera"

### Sensor Data and Monitoring

#### Environmental Monitoring
Get sensor readings and environmental data:

**Voice Commands**:
- "What's the temperature in the living room?"
- "Check the humidity in the basement"
- "Is anyone home?"

#### Energy Monitoring
Monitor energy usage and smart meters:

**Voice Commands**:
- "How much energy are we using?"
- "What's our current electricity cost?"
- "Show me the solar panel output"

#### Security Monitoring
Check security sensors and cameras:

**Voice Commands**:
- "Are all windows closed?"
- "Check the motion sensors"
- "Show me the security camera feeds"

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
Error: Entity 'light.nonexistent' not found
```

**Solutions**:
1. **Check entity ID** in Home Assistant Developer Tools
2. **Verify entity domain** (light, switch, etc.)
3. **Check entity availability**

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
- [Voice Audio Configuration](Voice-Audio.md) - Speech integration setup
- [Tool Development](Tool-Development.md) - Creating custom Home Assistant tools
- [External Services](External-Services.md) - Integrating other smart home platforms