import requests
import json
from datetime import datetime, timedelta
from typing import Optional
from utils import config
import pytz
from wiki_tools import get_wikipedia_with_context,query_wikipedia

def get_weather_forecast(location: str, date: Optional[str] = None, api_key: str = None) -> str:
    """
    Get weather forecast and astronomy data from weatherapi.com
    
    Args:
        location: City name, ZIP code, or coordinates (e.g., "Edgewood, WA" or "98372" or "47.25,-122.29")
        date: Date in YYYY-MM-DD format (optional, defaults to today)
        api_key: WeatherAPI.com API key (required)
    
    Returns:
        Formatted text response with weather and astronomy information
    """
    
    if not api_key:
        api_key = config.WEATHER_API_KEY
    
    # Get the local timezone from config
    local_tz = pytz.timezone(config.TIMEZONE)
 
    base_url = "https://api.weatherapi.com/v1"
    
    if date:
        try:
            # Parse the date string as a naive date
            target_date = datetime.strptime(date, "%Y-%m-%d").date()
            
            # Get current time in the configured timezone
            now_local = datetime.now(local_tz)
            today = now_local.date()
            
            days_ahead = (target_date - today).days
            
            if days_ahead <= 14 and days_ahead >= 0:
                url = f"{base_url}/forecast.json"
                params = {
                    "key": api_key,
                    "q": location,
                    "days": min(days_ahead + 1, 14),
                    "dt": date
                }
            elif days_ahead > 14 and days_ahead <= 365:
                # use future.json for dates 15-365 days out
                url = f"{base_url}/future.json"
                params = {
                    "key": api_key,
                    "q": location,
                    "dt": date
                }
            else:
                return f"Error: Date must be between today and 365 days in the future"
        except ValueError:
            return f"Error: Invalid date format. Please use YYYY-MM-DD"
    else:
        # use forecast.json for upcoming/today's weather
        url = f"{base_url}/forecast.json"
        params = {
            "key": api_key,
            "q": location,
            "days": 1
        }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        loc = data.get("location", {})
        location_info = f"{loc.get('name', 'Unknown')}, {loc.get('region', '')}, {loc.get('country', '')}"
        
        forecast_days = data.get("forecast", {}).get("forecastday", [])
        
        if not forecast_days:
            return f"No forecast data available for {location_info}"
        
        day_data = None
        if date:
            for day in forecast_days:
                if day.get("date") == date:
                    day_data = day
                    break
            if not day_data:
                day_data = forecast_days[0]
        else:
            day_data = forecast_days[0]
        
        day = day_data.get("day", {})
        astro = day_data.get("astro", {})
        
        if "hour" in day_data:
            del day_data["hour"]
        
        response_text = f"""Weather forecast for {location_info}
Date: {day_data.get('date', 'Unknown')}

Temperature:
  High: {day.get('maxtemp_f', 'N/A')}°F ({day.get('maxtemp_c', 'N/A')}°C)
  Low: {day.get('mintemp_f', 'N/A')}°F ({day.get('mintemp_c', 'N/A')}°C)
  Average: {day.get('avgtemp_f', 'N/A')}°F ({day.get('avgtemp_c', 'N/A')}°C)

Conditions: {day.get('condition', {}).get('text', 'Unknown')}
Precipitation: {day.get('totalprecip_in', 0)} in ({day.get('totalprecip_mm', 0)} mm)
Humidity: {day.get('avghumidity', 'N/A')}%
Max Wind: {day.get('maxwind_mph', 'N/A')} mph ({day.get('maxwind_kph', 'N/A')} kph)
Visibility: {day.get('avgvis_miles', 'N/A')} miles ({day.get('avgvis_km', 'N/A')} km)
UV Index: {day.get('uv', 'N/A')}

Astronomy:
  Sunrise: {astro.get('sunrise', 'N/A')}
  Sunset: {astro.get('sunset', 'N/A')}
  Moonrise: {astro.get('moonrise', 'N/A')}
  Moonset: {astro.get('moonset', 'N/A')}
  Moon Phase: {astro.get('moon_phase', 'N/A')}
  Moon Illumination: {astro.get('moon_illumination', 'N/A')}%"""
        
        return response_text
        
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except Exception as e:
        return f"Error processing weather data: {str(e)}"
