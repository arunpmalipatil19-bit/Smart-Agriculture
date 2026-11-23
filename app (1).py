# ============================================================
# SMART AGRICULTURE AI + IoT DECISION SUPPORT SYSTEM
# Complete Flask Backend
# ============================================================
#
# Project:
# AI and IoT-Based Agricultural Decision Support System
# Using Seasonal Weather Forecasting for Crop Recommendation
# and Precision Irrigation Planning
#
# Hardware:
# ESP-12E / ESP8266 + Capacitive Soil Moisture + DHT22 + Rain Sensor
#
# Backend:
# Flask + Pandas + NumPy + Requests + Joblib + Random Forest
#
# IMPORTANT:
# - The crop model used here is the trained deployment Random Forest.
# - The deployment model expects:
#       rainfall, humidity, temperature, ph
# - Seasonal rainfall is the sum of the first four monthly
#   precipitation_mean values returned by the Open-Meteo seasonal API.
# - Seasonal weather is an outlook/assessment, not an exact
#   four-month local daily prediction.
# ============================================================

import csv
import json
import math
import os
import re
import traceback
from io import BytesIO

try:
    from gtts import gTTS
except Exception:
    gTTS = None
from datetime import datetime, timezone
from urllib.parse import quote

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory, send_file


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
ML_DIR = os.path.join(PROJECT_DIR, "ml")
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

SOIL_CSV = os.path.join(DATA_DIR, "soil_ph_reference.csv")
CROP_REFERENCE_CSV = os.path.join(DATA_DIR, "crop_reference.csv")

MODEL_CANDIDATES = [
    os.path.join(ML_DIR, "crop_deployment_model.pkl"),
    os.path.join(ML_DIR, "crop_model.pkl"),
]

IOT_DATA_FILE = os.path.join(BASE_DIR, "latest_iot_data.json")


# ============================================================
# 2. FLASK APPLICATION
# ============================================================

app = Flask(__name__, static_folder=None)

STATE = "Karnataka"
SERVER_PORT = 5000


# ============================================================
# 3. EXTERNAL API ENDPOINTS
# ============================================================

OPEN_METEO_GEOCODING = (
    "https://geocoding-api.open-meteo.com/v1/search"
)

OPEN_METEO_FORECAST = (
    "https://api.open-meteo.com/v1/forecast"
)

OPEN_METEO_SEASONAL = (
    "https://seasonal-api.open-meteo.com/v1/seasonal"
)


# ============================================================
# 4. HTTP SESSION
# ============================================================

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Smart-Agriculture-Decision-Support/"
            "1.0 (educational project)"
        )
    }
)


def safe_float(value, default=None):
    """Convert a value to float without crashing the API."""
    try:
        if value is None:
            return default

        number = float(value)

        if not math.isfinite(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_int(value, default=None):
    """Convert a value to int without crashing the API."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def now_iso():
    """Return a simple UTC timestamp for IoT data."""
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# 5. DISTRICT / TALUK NAME NORMALIZATION
# ============================================================

DISTRICT_ALIASES = {
    "mysore": "mysuru",
    "mysuru": "mysuru",

    "belgaum": "belagavi",
    "belagavi": "belagavi",

    "bellary": "ballari",
    "ballari": "ballari",

    "bijapur": "vijayapura",
    "vijayapura": "vijayapura",

    "gulbarga": "kalaburagi",
    "kalaburagi": "kalaburagi",

    "shimoga": "shivamogga",
    "shivamogga": "shivamogga",

    "tumkur": "tumakuru",
    "tumakuru": "tumakuru",
}


TALUK_ALIASES = {
    "hagaribommanahalli": [
        "Hagaribommanahalli",
        "Hagari Bommanahalli",
        "Hagaribommanahalli Taluk",
    ],
    "hospet": [
        "Hospet",
        "Hosapete",
        "Vijayanagara",
    ],
    "hosapete": [
        "Hosapete",
        "Hospet",
    ],
    "chikmagalur": [
        "Chikmagalur",
        "Chikkamagaluru",
    ],
    "chikkamagaluru": [
        "Chikkamagaluru",
        "Chikmagalur",
    ],
    "hubli": [
        "Hubli",
        "Hubballi",
    ],
    "hubballi": [
        "Hubballi",
        "Hubli",
    ],
    "shimoga": [
        "Shimoga",
        "Shivamogga",
    ],
    "shivamogga": [
        "Shivamogga",
        "Shimoga",
    ],
    "tumkur": [
        "Tumkur",
        "Tumakuru",
    ],
    "tumakuru": [
        "Tumakuru",
        "Tumkur",
    ],
}


def clean_taluk(value):
    """
    Remove atlas codes such as:
    Badami (580) -> Badami
    """
    if value is None:
        return ""

    value = str(value).strip()

    value = re.sub(
        r"\s*\([^)]*\)",
        "",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def clean_name(value):
    """Create a comparison-safe lowercase name."""
    value = clean_taluk(value).lower()

    value = value.replace("&", "and")

    value = re.sub(
        r"[^a-z0-9]+",
        " ",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def canonical_district(value):
    """Convert old/new district names to one comparison key."""
    key = clean_name(value)

    return DISTRICT_ALIASES.get(
        key,
        key,
    )


def taluk_search_names(taluk):
    """Return spelling variants for Open-Meteo geocoding."""
    cleaned = clean_taluk(taluk)
    key = clean_name(cleaned)

    names = []

    for name in [cleaned] + TALUK_ALIASES.get(key, []):
        if name and name not in names:
            names.append(name)

    return names


# ============================================================
# 6. SOIL DATA
# ============================================================

soil_rows = []


def load_soil_data():
    """Load the 165 taluk-level soil-pH reference records."""
    global soil_rows

    soil_rows = []

    if not os.path.exists(SOIL_CSV):
        print()
        print("WARNING: Soil CSV not found:")
        print(SOIL_CSV)
        print()
        return

    try:
        df = pd.read_csv(SOIL_CSV)

        print("Soil CSV columns:", list(df.columns))

        # Accept small column-name variations.
        district_col = next(
            (
                col
                for col in df.columns
                if clean_name(col) == "district"
            ),
            None,
        )

        taluk_col = next(
            (
                col
                for col in df.columns
                if clean_name(col) == "taluk"
            ),
            None,
        )

        ph_col = next(
            (
                col
                for col in df.columns
                if clean_name(col) in {
                    "soil ph",
                    "ph",
                    "soil ph value",
                }
            ),
            None,
        )

        if district_col is None:
            raise ValueError(
                "Column 'district' is missing from soil_ph_reference.csv"
            )

        if taluk_col is None:
            raise ValueError(
                "Column 'taluk' is missing from soil_ph_reference.csv"
            )

        if ph_col is None:
            raise ValueError(
                "Column 'soil_ph' is missing from soil_ph_reference.csv"
            )

        for _, row in df.iterrows():

            district = str(
                row.get(district_col, "")
            ).strip()

            taluk = clean_taluk(
                row.get(taluk_col, "")
            )

            soil_ph = safe_float(
                row.get(ph_col)
            )

            if (
                district
                and taluk
                and soil_ph is not None
            ):
                soil_rows.append(
                    {
                        "district": district,
                        "taluk": taluk,
                        "district_key": canonical_district(
                            district
                        ),
                        "taluk_key": clean_name(
                            taluk
                        ),
                        "samples": row.get(
                            "samples",
                            None,
                        ),
                        "soil_ph": soil_ph,
                    }
                )

        print(
            "Soil records loaded:",
            len(soil_rows),
        )

    except Exception as exc:
        print("ERROR loading soil CSV:", exc)
        traceback.print_exc()
        soil_rows = []


load_soil_data()


def find_soil_record(district, taluk):
    """Find exact district + taluk soil reference."""
    district_key = canonical_district(district)
    taluk_key = clean_name(taluk)

    for row in soil_rows:
        if (
            row["district_key"] == district_key
            and row["taluk_key"] == taluk_key
        ):
            return row

    # Safe fallback: taluk-only match.
    # Useful when the atlas and UI use an old district spelling.
    for row in soil_rows:
        if row["taluk_key"] == taluk_key:
            return row

    return None


# ============================================================
# 7. OPTIONAL LOCAL CROP CONTEXT
# ============================================================

crop_reference_rows = []


def load_crop_reference():
    """
    Optional local crop-context file.

    This file is NOT used to override the ML prediction.
    It is only returned as supporting local agricultural context.
    """
    global crop_reference_rows

    crop_reference_rows = []

    if not os.path.exists(CROP_REFERENCE_CSV):
        return

    try:
        with open(
            CROP_REFERENCE_CSV,
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            for row in reader:
                crop_reference_rows.append(
                    {
                        key: (
                            value.strip()
                            if isinstance(value, str)
                            else value
                        )
                        for key, value in row.items()
                    }
                )

        print(
            "Crop reference records:",
            len(crop_reference_rows),
        )

    except Exception as exc:
        print(
            "Crop reference warning:",
            exc,
        )
        crop_reference_rows = []


load_crop_reference()


def find_crop_reference(district, taluk):
    """Return optional local crop context."""
    district_key = canonical_district(district)
    taluk_key = clean_name(taluk)

    for row in crop_reference_rows:

        row_district = canonical_district(
            row.get("district", "")
        )

        row_taluk = clean_name(
            row.get("taluk", "")
        )

        if (
            row_district == district_key
            and row_taluk == taluk_key
        ):
            return row

    for row in crop_reference_rows:

        if (
            clean_name(row.get("taluk", ""))
            == taluk_key
        ):
            return row

    return None


# ============================================================
# 8. RANDOM FOREST MODEL
# ============================================================

crop_model = None
model_path_used = None
model_error = None


def load_crop_model():
    """Load the deployment Random Forest model."""
    global crop_model
    global model_path_used
    global model_error

    crop_model = None
    model_path_used = None
    model_error = None

    for path in MODEL_CANDIDATES:

        if not os.path.exists(path):
            continue

        try:
            model = joblib.load(path)

            # The deployment model should accept exactly:
            # rainfall, humidity, temperature, ph
            feature_names = list(
                getattr(
                    model,
                    "feature_names_in_",
                    [],
                )
            )

            if feature_names:
                normalized = [
                    clean_name(name)
                    for name in feature_names
                ]

                expected = {
                    "rainfall",
                    "humidity",
                    "temperature",
                    "ph",
                }

                if not expected.issubset(
                    set(normalized)
                ):
                    # crop_model.pkl is normally the 7-feature
                    # training model, not the deployment model.
                    if os.path.basename(path) == (
                        "crop_model.pkl"
                    ):
                        continue

            crop_model = model
            model_path_used = path

            print(
                "AI MODEL LOADED:",
                path,
            )

            print(
                "Model type:",
                type(model).__name__,
            )

            if hasattr(model, "feature_names_in_"):
                print(
                    "Model features:",
                    list(model.feature_names_in_),
                )

            return

        except Exception as exc:
            model_error = str(exc)
            print(
                "Model load failed:",
                path,
                exc,
            )

    print()
    print(
        "WARNING: No compatible deployment model was loaded."
    )
    print(
        "Expected:",
        os.path.join(
            ML_DIR,
            "crop_deployment_model.pkl",
        ),
    )
    print()


load_crop_model()


def model_feature_columns():
    """
    Return model feature names if available.
    The deployment model should use:
    rainfall, humidity, temperature, ph.
    """
    if crop_model is None:
        return []

    names = getattr(
        crop_model,
        "feature_names_in_",
        None,
    )

    if names is not None:
        return [
            str(name)
            for name in names
        ]

    return [
        "rainfall",
        "humidity",
        "temperature",
        "ph",
    ]


def build_model_input(
    rainfall,
    humidity,
    temperature,
    soil_ph,
):
    """
    Build the exact feature DataFrame expected by the
    deployment model.
    """
    values = {
        "rainfall": rainfall,
        "humidity": humidity,
        "temperature": temperature,
        "ph": soil_ph,
    }

    columns = model_feature_columns()

    if not columns:
        columns = [
            "rainfall",
            "humidity",
            "temperature",
            "ph",
        ]

    # Ensure the model only receives the known four features.
    normalized_to_value = {
        "rainfall": rainfall,
        "humidity": humidity,
        "temperature": temperature,
        "ph": soil_ph,
    }

    data = []

    for column in columns:
        key = clean_name(column)

        if key not in normalized_to_value:
            raise ValueError(
                "Loaded model requires unsupported feature "
                f"'{column}'. Use crop_deployment_model.pkl."
            )

        data.append(
            normalized_to_value[key]
        )

    return pd.DataFrame(
        [data],
        columns=columns,
    )


# ============================================================
# 9. IoT DATA STORAGE
# ============================================================

DEFAULT_IOT_DATA = {
    "soil_moisture": None,
    "temperature": None,
    "humidity": None,
    "rain": None,
    "updated_at": None,
}


def normalize_iot_data(data):
    """Validate and normalize incoming ESP8266 data."""
    if not isinstance(data, dict):
        raise ValueError(
            "IoT payload must be a JSON object."
        )

    soil_moisture = safe_float(
        data.get("soil_moisture")
    )

    temperature = safe_float(
        data.get("temperature")
    )

    humidity = safe_float(
        data.get("humidity")
    )

    rain = data.get("rain")

    if rain is not None:
        rain = str(rain).strip()

    if soil_moisture is not None:
        soil_moisture = max(
            0.0,
            min(100.0, soil_moisture),
        )

    if humidity is not None:
        humidity = max(
            0.0,
            min(100.0, humidity),
        )

    return {
        "soil_moisture": soil_moisture,
        "temperature": temperature,
        "humidity": humidity,
        "rain": rain,
        "updated_at": now_iso(),
    }


def load_iot_data():
    """Read the latest ESP8266 values."""
    if not os.path.exists(IOT_DATA_FILE):
        return DEFAULT_IOT_DATA.copy()

    try:
        with open(
            IOT_DATA_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        result = DEFAULT_IOT_DATA.copy()
        result.update(data)

        return result

    except Exception as exc:
        print(
            "IoT data read warning:",
            exc,
        )
        return DEFAULT_IOT_DATA.copy()


def save_iot_data(data):
    """Write latest ESP8266 values to disk."""
    os.makedirs(
        os.path.dirname(IOT_DATA_FILE),
        exist_ok=True,
    )

    with open(
        IOT_DATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# 10. KARNATAKA LOCATION RESOLUTION
# ============================================================

# These coordinates are representative points for the
# selected taluk/urban area.
#
# They are used for weather and seasonal API requests.
# Open-Meteo accepts latitude/longitude directly, so these
# locations do not depend on online administrative geocoding.

LOCATION_OVERRIDES = {

    # ========================================================
    # BENGALURU URBAN
    # ========================================================

    (
        "bengaluru urban",
        "bengaluru anekal urban",
    ): {
        "latitude": 12.7111,
        "longitude": 77.6956,
        "name": "Anekal",
    },

    (
        "bengaluru urban",
        "anekal",
    ): {
        "latitude": 12.7111,
        "longitude": 77.6956,
        "name": "Anekal",
    },

    (
        "bengaluru urban",
        "bengaluru north",
    ): {
        "latitude": 13.1007,
        "longitude": 77.5963,
        "name": "Bengaluru North",
    },

    (
        "bengaluru urban",
        "bengaluru east",
    ): {
        "latitude": 13.0041,
        "longitude": 77.7043,
        "name": "Bengaluru East",
    },

    (
        "bengaluru urban",
        "bengaluru south",
    ): {
        "latitude": 12.8851,
        "longitude": 77.4969,
        "name": "Bengaluru South",
    },

    # ========================================================
    # DAKSHINA KANNADA
    # ========================================================

    (
        "dakshina kannada",
        "bantwal",
    ): {
        "latitude": 12.8900,
        "longitude": 75.0340,
        "name": "Bantwal",
    },

    (
        "dakshina kannada",
        "belthangady",
    ): {
        "latitude": 12.9840,
        "longitude": 75.2970,
        "name": "Belthangady",
    },

    (
        "dakshina kannada",
        "mangalore",
    ): {
        "latitude": 12.9141,
        "longitude": 74.8560,
        "name": "Mangalore",
    },

    (
        "dakshina kannada",
        "mangaluru",
    ): {
        "latitude": 12.9141,
        "longitude": 74.8560,
        "name": "Mangalore",
    },

    (
        "dakshina kannada",
        "puttur",
    ): {
        "latitude": 12.7596,
        "longitude": 75.2049,
        "name": "Puttur",
    },

    (
        "dakshina kannada",
        "sulya",
    ): {
        "latitude": 12.5616,
        "longitude": 75.3872,
        "name": "Sullia",
    },

    (
        "dakshina kannada",
        "sullia",
    ): {
        "latitude": 12.5616,
        "longitude": 75.3872,
        "name": "Sullia",
    },
}


def get_location_override(district, taluk):

    district_key = clean_name(district)
    taluk_key = clean_name(taluk)

    return LOCATION_OVERRIDES.get(
        (
            district_key,
            taluk_key,
        )
    )


def geocode_location(district, taluk):

    # ========================================================
    # 1. STATIC KARNATAKA LOCATION
    # ========================================================

    override = get_location_override(
        district,
        taluk,
    )

    if override is not None:

        print(
            "[LOCATION] Static coordinate used:"
        )

        print(
            f"District : {district}"
        )

        print(
            f"Taluk    : {taluk}"
        )

        print(
            f"Latitude : {override['latitude']}"
        )

        print(
            f"Longitude: {override['longitude']}"
        )

        return {
            "latitude": override["latitude"],
            "longitude": override["longitude"],
            "name": override["name"],
            "source": "Karnataka coordinate reference",
            "query": (
                f"{taluk}, {district}, Karnataka, India"
            ),
        }

    # ========================================================
    # 2. ONLINE GEOCODING FOR OTHER TALUKS
    # ========================================================

    district_text = clean_taluk(district)

    taluk_names = taluk_search_names(
        taluk
    )

    queries = []

    for name in taluk_names:

        queries.append(
            (
                name,
                f"{name}, {district_text}, Karnataka, India",
                "taluk+district",
            )
        )

    for name in taluk_names:

        queries.append(
            (
                name,
                f"{name}, Karnataka, India",
                "taluk",
            )
        )

    queries.append(
        (
            district_text,
            f"{district_text}, Karnataka, India",
            "district",
        )
    )

    seen = set()

    for _, query, source_type in queries:

        if query in seen:
            continue

        seen.add(query)

        try:

            response = session.get(
                OPEN_METEO_GEOCODING,
                params={
                    "name": query,
                    "count": 10,
                    "language": "en",
                    "format": "json",
                    "countryCode": "IN",
                },
                timeout=15,
            )

            response.raise_for_status()

            results = response.json().get(
                "results",
                []
            )

            if not results:
                continue

            selected = None

            # ------------------------------------------------
            # Prefer Karnataka result
            # ------------------------------------------------

            for result in results:

                country = clean_name(
                    result.get(
                        "country",
                        "",
                    )
                )

                admin1 = clean_name(
                    result.get(
                        "admin1",
                        "",
                    )
                )

                result_name = clean_name(
                    result.get(
                        "name",
                        "",
                    )
                )

                if (
                    country == "india"
                    and "karnataka" in admin1
                    and any(
                        clean_name(name)
                        == result_name
                        for name in taluk_names
                    )
                ):

                    selected = result

                    break

            # ------------------------------------------------
            # Any Karnataka result
            # ------------------------------------------------

            if selected is None:

                for result in results:

                    country = clean_name(
                        result.get(
                            "country",
                            "",
                        )
                    )

                    admin1 = clean_name(
                        result.get(
                            "admin1",
                            "",
                        )
                    )

                    if (
                        country == "india"
                        and "karnataka" in admin1
                    ):

                        selected = result

                        break

            if selected is None:

                selected = results[0]

            latitude = safe_float(
                selected.get(
                    "latitude"
                )
            )

            longitude = safe_float(
                selected.get(
                    "longitude"
                )
            )

            if (
                latitude is None
                or longitude is None
            ):

                continue

            return {
                "latitude": latitude,
                "longitude": longitude,
                "name": selected.get(
                    "name",
                    taluk,
                ),
                "source": (
                    "taluk geocoding"
                    if source_type != "district"
                    else "district fallback"
                ),
                "query": query,
            }

        except Exception as exc:

            print(
                "[GEOCODING WARNING]",
                query,
                repr(exc),
            )

    return None


# ============================================================
# 11. WEATHER CODE DESCRIPTION
# ============================================================

WEATHER_CODES = {

    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Depositing rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    66: "Light freezing rain",
    67: "Heavy freezing rain",

    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",

    77: "Snow grains",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    85: "Slight snow showers",
    86: "Heavy snow showers",

    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def weather_code_description(code):

    code = safe_int(
        code
    )

    if code is None:
        return "Unknown"

    return WEATHER_CODES.get(
        code,
        "Unknown",
    )


# ============================================================
# 12. CURRENT + SHORT-TERM WEATHER
# ============================================================

def get_weather_data(
    latitude,
    longitude,
):

    print()
    print(
        "========== WEATHER REQUEST =========="
    )

    print(
        "Latitude :",
        latitude,
    )

    print(
        "Longitude:",
        longitude,
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "weather_code"
        ),

        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "weather_code"
        ),

        "forecast_days": 7,

        "timezone": "auto",
    }

    # ========================================================
    # PRIMARY REQUEST
    # ========================================================

    try:

        response = session.get(
            OPEN_METEO_FORECAST,
            params=params,
            timeout=30,
        )

        print(
            "Weather HTTP status:",
            response.status_code,
        )

        response.raise_for_status()

        data = response.json()

    except Exception as first_error:

        print(
            "[WEATHER PRIMARY ERROR]",
            repr(first_error),
        )

        # ====================================================
        # SECOND REQUEST
        # ====================================================
        # Use a fresh requests connection instead of the
        # persistent session.

        try:

            response = requests.get(
                OPEN_METEO_FORECAST,
                params=params,
                headers={
                    "User-Agent": (
                        "Smart-Agriculture-Decision-Support/1.0"
                    )
                },
                timeout=30,
            )

            print(
                "Weather retry HTTP status:",
                response.status_code,
            )

            response.raise_for_status()

            data = response.json()

        except Exception as second_error:

            print(
                "[WEATHER RETRY ERROR]",
                repr(second_error),
            )

            raise RuntimeError(
                "Open-Meteo weather request failed. "
                f"Primary: {first_error}; "
                f"Retry: {second_error}"
            )


    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    current = data.get(
        "current",
        {},
    )

    temperature = safe_float(
        current.get(
            "temperature_2m"
        )
    )

    humidity = safe_float(
        current.get(
            "relative_humidity_2m"
        )
    )

    precipitation = safe_float(
        current.get(
            "precipitation"
        ),
        0.0,
    )

    wind_speed = safe_float(
        current.get(
            "wind_speed_10m"
        )
    )

    weather_code = safe_int(
        current.get(
            "weather_code"
        )
    )

    if temperature is None:
        raise RuntimeError(
            "Weather API returned no temperature."
        )

    if humidity is None:
        raise RuntimeError(
            "Weather API returned no humidity."
        )


    # ========================================================
    # 7-DAY WEATHER
    # ========================================================

    daily = data.get(
        "daily",
        {},
    )

    dates = daily.get(
        "time",
        [],
    )

    max_values = daily.get(
        "temperature_2m_max",
        [],
    )

    min_values = daily.get(
        "temperature_2m_min",
        [],
    )

    precipitation_values = daily.get(
        "precipitation_sum",
        [],
    )

    probability_values = daily.get(
        "precipitation_probability_max",
        [],
    )

    code_values = daily.get(
        "weather_code",
        [],
    )


    daily_forecast = []


    for index, date in enumerate(
        dates
    ):

        code = (
            code_values[index]
            if index < len(code_values)
            else None
        )

        daily_forecast.append(
            {
                "date": date,

                "temperature_max": (
                    safe_float(
                        max_values[index]
                    )
                    if index < len(max_values)
                    else None
                ),

                "temperature_min": (
                    safe_float(
                        min_values[index]
                    )
                    if index < len(min_values)
                    else None
                ),

                "precipitation": (
                    safe_float(
                        precipitation_values[index]
                    )
                    if index < len(
                        precipitation_values
                    )
                    else None
                ),

                "precipitation_probability": (
                    safe_float(
                        probability_values[index]
                    )
                    if index < len(
                        probability_values
                    )
                    else None
                ),

                "weather_code": code,

                "condition":
                    weather_code_description(
                        code
                    ),
            }
        )


    print(
        "Temperature:",
        temperature,
    )

    print(
        "Humidity:",
        humidity,
    )

    print(
        "Rain:",
        precipitation,
    )

    print(
        "Wind:",
        wind_speed,
    )

    print(
        "Condition:",
        weather_code_description(
            weather_code
        ),
    )

    print(
        "====================================="
    )


    return {

        "temperature":
            temperature,

        "humidity":
            humidity,

        "rain":
            precipitation,

        "wind_speed":
            wind_speed,

        "weather_code":
            weather_code,

        "condition":
            weather_code_description(
                weather_code
            ),

        "daily":
            daily_forecast,
    }


# ============================================================
# 13. SEASONAL WEATHER SERVICE
# ============================================================

# ============================================================
# 13. SEASONAL WEATHER SERVICE
# ============================================================

def get_seasonal_data(latitude, longitude):
    """
    Get a four-month seasonal weather assessment.

    IMPORTANT:
    This is a probabilistic seasonal outlook from ECMWF through
    Open-Meteo. It is NOT an exact four-month local daily forecast.

    Open-Meteo supports these seasonal ensemble-mean models:
      - ecmwf_seasonal_ensemble_mean_seamless
      - ecmwf_seas5_ensemble_mean

    Monthly variables used by the dashboard:
      - temperature_2m_mean
      - temperature_2m_anomaly
      - precipitation_mean
      - precipitation_anomaly
    """

    monthly_variables = (
        "temperature_2m_mean,"
        "temperature_2m_anomaly,"
        "precipitation_mean,"
        "precipitation_anomaly"
    )

    # First choice: seamless EC46 + SEAS5 ensemble mean.
    # Second choice: SEAS5 ensemble mean.
    models_to_try = [
        "ecmwf_seasonal_ensemble_mean_seamless",
        "ecmwf_seas5_ensemble_mean",
    ]

    last_error = None

    for model_name in models_to_try:
        try:
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "models": model_name,
                "monthly": monthly_variables,
                "forecast_days": 217,
                "cell_selection": "land",
                "timezone": "auto",
            }

            print("SEASONAL REQUEST MODEL:", model_name)
            print("SEASONAL REQUEST PARAMS:", params)

            response = session.get(
                OPEN_METEO_SEASONAL,
                params=params,
                timeout=45,
            )

            print(
                "SEASONAL HTTP STATUS:",
                response.status_code,
                "MODEL:",
                model_name,
            )

            response.raise_for_status()

            data = response.json()

            # Open-Meteo returns monthly data as an object of arrays.
            monthly = data.get("monthly") or {}

            time_values = monthly.get("time") or []
            temp_mean = monthly.get("temperature_2m_mean") or []
            temp_anomaly = monthly.get("temperature_2m_anomaly") or []
            precipitation_mean = monthly.get("precipitation_mean") or []
            precipitation_anomaly = monthly.get("precipitation_anomaly") or []

            if not time_values:
                raise ValueError(
                    "Seasonal API returned no monthly dates."
                )

            if not precipitation_mean:
                raise ValueError(
                    "Seasonal API returned no monthly precipitation values."
                )

            monthly_rows = []
            count = min(4, len(time_values))

            for index in range(count):
                monthly_rows.append({
                    "month": time_values[index],
                    "temperature_mean": (
                        safe_float(temp_mean[index])
                        if index < len(temp_mean)
                        else None
                    ),
                    "temperature_anomaly": (
                        safe_float(temp_anomaly[index])
                        if index < len(temp_anomaly)
                        else None
                    ),
                    "precipitation_mean": (
                        safe_float(precipitation_mean[index])
                        if index < len(precipitation_mean)
                        else None
                    ),
                    "precipitation_anomaly": (
                        safe_float(precipitation_anomaly[index])
                        if index < len(precipitation_anomaly)
                        else None
                    ),
                })

            rainfall_values = [
                row["precipitation_mean"]
                for row in monthly_rows
                if row["precipitation_mean"] is not None
            ]

            if not rainfall_values:
                raise ValueError(
                    "No usable seasonal precipitation values."
                )

            four_month_rainfall = round(
                sum(rainfall_values),
                1,
            )

            print(
                "SEASONAL SUCCESS:",
                model_name,
                "MONTHS:",
                len(monthly_rows),
                "4-MONTH RAINFALL:",
                four_month_rainfall,
            )

            return {
                "monthly": monthly_rows,
                "four_month_rainfall": four_month_rainfall,
                "source": "Open-Meteo ECMWF seasonal ensemble-mean forecast",
                "model": model_name,
                "note": (
                    "Probabilistic seasonal planning outlook; "
                    "not an exact local daily prediction."
                ),
            }

        except Exception as error:
            print(
                "SEASONAL API ERROR:",
                repr(error),
                "MODEL:",
                model_name,
            )
            last_error = error

    raise RuntimeError(
        "Seasonal weather data could not be obtained. "
        f"Last error: {last_error}"
    )


# ============================================================
# 14. LOCATION DATA HELPER
# ============================================================

def get_location_for_area(district, taluk):
    location = geocode_location(
        district,
        taluk,
    )

    if location is None:
        raise RuntimeError(
            "Unable to locate the selected Karnataka "
            "district/taluk using the weather service."
        )

    return location


# ============================================================
# 15. FRONTEND ROUTES
# ============================================================

@app.after_request
def add_no_cache_headers(response):
    # Always load the latest frontend JS/CSS during development and expo demo.
    if request.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.route("/")
def home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html",
    )


@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(
        FRONTEND_DIR,
        filename,
    )


# ============================================================
# 16. HEALTH API
# ============================================================

@app.route(
    "/api/health",
    methods=["GET"],
)
def health():
    return jsonify(
        {
            "status": "success",
            "message": (
                "Smart Agriculture AI + IoT Backend Running"
            ),
            "state": STATE,
            "soil_records": len(soil_rows),
            "crop_reference_records": len(
                crop_reference_rows
            ),
            "ai_model": (
                "LOADED"
                if crop_model is not None
                else "NOT LOADED"
            ),
            "model_file": model_path_used,
            "model_error": model_error,
            "iot": "ENABLED",
            "routes": [
                "/",
                "/api/health",
                "/api/taluks/<district>",
                "/api/soil/<district>/<taluk>",
                "/api/weather/<district>/<taluk>",
                "/api/seasonal/<district>/<taluk>",
                "/api/recommend-crop/<district>/<taluk>",
                "/api/iot-data",
                "/api/irrigation/<district>/<taluk>",
                "/api/tts",
            ],
        }
    )


# ============================================================
# 17. TALUK API
# ============================================================

@app.route(
    "/api/taluks/<district>",
    methods=["GET"],
)
def get_taluks(district):
    district_key = canonical_district(
        district
    )

    taluks = []

    for row in soil_rows:

        if (
            row["district_key"]
            == district_key
        ):
            if row["taluk"] not in taluks:
                taluks.append(
                    row["taluk"]
                )

    taluks.sort(
        key=lambda item: clean_name(item)
    )

    if not taluks:
        return jsonify(
            {
                "status": "error",
                "error": (
                    "No taluks found for selected district."
                ),
                "district": district,
            }
        ), 404

    return jsonify(
        {
            "status": "success",
            "state": STATE,
            "district": district,
            "taluks": taluks,
        }
    )


# ============================================================
# 18. SOIL API
# ============================================================

@app.route(
    "/api/soil/<district>/<taluk>",
    methods=["GET"],
)
def get_soil(district, taluk):

    record = find_soil_record(
        district,
        taluk,
    )

    if record is None:
        return jsonify(
            {
                "status": "error",
                "error": (
                    "Soil information not found "
                    "for the selected taluk."
                ),
            }
        ), 404

    return jsonify(
        {
            "status": "success",
            "state": STATE,
            "district": record["district"],
            "taluk": record["taluk"],
            "soil_ph": record["soil_ph"],
            "samples": record["samples"],
            "source": (
                "Karnataka Soil Fertility Atlas reference data"
            ),
        }
    )


# ============================================================
# 19. WEATHER API
# ============================================================

@app.route(
    "/api/weather/<district>/<taluk>",
    methods=["GET"],
)

def weather_route(district, taluk):

    try:

        # ----------------------------------------------------
        # Get coordinates
        # ----------------------------------------------------

        location = get_location_for_area(
            district,
            taluk,
        )

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # Get Open-Meteo weather
        # ----------------------------------------------------

        weather = get_weather_data(
            latitude,
            longitude,
        )

        return jsonify(
            {
                "status": "success",

                "state": STATE,

                "district": district,

                "taluk": taluk,

                "location": location["name"],

                "location_source":
                    location["source"],

                "temperature":
                    weather["temperature"],

                "humidity":
                    weather["humidity"],

                "rain":
                    weather["rain"],

                "wind_speed":
                    weather["wind_speed"],

                "weather_code":
                    weather["weather_code"],

                "condition":
                    weather["condition"],

                "daily":
                    weather["daily"],

                "source":
                    "Open-Meteo weather forecast",
            }
        )

    except Exception as exc:

        print()
        print(
            "WEATHER API ERROR:",
            repr(exc),
        )

        traceback.print_exc()

        # ----------------------------------------------------
        # IMPORTANT:
        # Do NOT crash the complete dashboard.
        #
        # Use ESP8266 DHT22 as local field fallback when
        # available.
        # ----------------------------------------------------

        iot = load_iot_data()

        iot_temperature = safe_float(
            iot.get("temperature")
        )

        iot_humidity = safe_float(
            iot.get("humidity")
        )

        iot_rain = iot.get(
            "rain"
        )

        if (
            iot_temperature is not None
            and
            iot_humidity is not None
        ):

            return jsonify(
                {
                    "status": "success",

                    "state": STATE,

                    "district": district,

                    "taluk": taluk,

                    "temperature":
                        iot_temperature,

                    "humidity":
                        iot_humidity,

                    "rain":
                        0.0,

                    "wind_speed":
                        None,

                    "weather_code":
                        None,

                    "condition":
                        (
                            "Live field weather "
                            "(ESP8266)"
                        ),

                    "daily":
                        [],

                    "source":
                        (
                            "ESP8266 DHT22 "
                            "fallback"
                        ),

                    "warning":
                        (
                            "Online weather API "
                            "temporarily unavailable. "
                            "Showing live field "
                            "temperature and humidity."
                        ),
                }
            )

        # ----------------------------------------------------
        # If ESP data is also unavailable, return a clear
        # service status instead of crashing the page.
        # ----------------------------------------------------

        return jsonify(
            {
                "status": "success",

                "state": STATE,

                "district": district,

                "taluk": taluk,

                "temperature": None,

                "humidity": None,

                "rain": None,

                "wind_speed": None,

                "weather_code": None,

                "condition":
                    "Weather temporarily unavailable",

                "daily": [],

                "source":
                    "Weather API unavailable",

                "warning":
                    (
                        "Live weather could not be "
                        "obtained at this moment."
                    ),

                "details":
                    str(exc),
            }
        )

# ============================================================
# 20. SEASONAL WEATHER API
# ============================================================

@app.route(
    "/api/seasonal/<district>/<taluk>",
    methods=["GET"],
    strict_slashes=False,
)
def seasonal_route(district, taluk):
    """Return the four-month seasonal assessment for the selected taluk."""
    try:
        location = get_location_for_area(district, taluk)

        seasonal = get_seasonal_data(
            location["latitude"],
            location["longitude"],
        )

        rows = seasonal.get("monthly", [])

        # Keep ONE canonical frontend format: monthly rows.
        # Also expose the array fields for easy debugging/compatibility.
        monthly_rows = rows[:4]

        monthly_arrays = {
            "time": [row.get("month") for row in monthly_rows],
            "temperature_2m_mean": [
                row.get("temperature_mean")
                for row in monthly_rows
            ],
            "temperature_2m_anomaly": [
                row.get("temperature_anomaly")
                for row in monthly_rows
            ],
            "precipitation_mean": [
                row.get("precipitation_mean")
                for row in monthly_rows
            ],
            "precipitation_anomaly": [
                row.get("precipitation_anomaly")
                for row in monthly_rows
            ],
        }

        return jsonify({
            "status": "success",
            "state": STATE,
            "district": district,
            "taluk": taluk,
            "location": location["name"],
            "location_source": location["source"],
            "monthly": monthly_rows,
            "monthly_arrays": monthly_arrays,
            "four_month_rainfall": seasonal["four_month_rainfall"],
            "source": seasonal["source"],
            "model": seasonal["model"],
            "note": seasonal["note"],
        })

    except Exception as exc:
        print("SEASONAL API ROUTE ERROR:", repr(exc))
        traceback.print_exc()

        # This is intentionally an error response.
        # We do NOT pretend that short-term rainfall is a four-month
        # seasonal value when the seasonal service is unavailable.
        return jsonify({
            "status": "error",
            "state": STATE,
            "district": district,
            "taluk": taluk,
            "location": taluk,
            "location_source": "Unavailable",
            "monthly": [],
            "monthly_arrays": {
                "time": [],
                "temperature_2m_mean": [],
                "temperature_2m_anomaly": [],
                "precipitation_mean": [],
                "precipitation_anomaly": [],
            },
            "four_month_rainfall": None,
            "source": "Seasonal service unavailable",
            "model": None,
            "note": (
                "Seasonal outlook is unavailable right now. "
                "No short-term rainfall value is being substituted."
            ),
            "error": str(exc),
        }), 503


# ============================================================
# 21. HYBRID CROP RECOMMENDATION
# ============================================================

# IMPORTANT DESIGN NOTE
# ---------------------
# The original 4-feature Random Forest was trained on a generic crop
# recommendation dataset. That dataset does not contain Karnataka
# location/cropping information and contains only four deployment
# features: rainfall, humidity, temperature and pH.
#
# If we feed a Karnataka taluk's live weather directly into that model,
# crops such as papaya/coconut can become the winner even when they are
# not the dominant agricultural choices for that taluk.
#
# Therefore the recommendation below is intentionally HYBRID:
#   1) Random Forest = climate/soil ML evidence
#   2) Karnataka crop prior = local agricultural context
#   3) The two are combined into a suitability score
#
# The Random Forest is NEVER replaced by a hard-coded crop. Local context
# only re-weights the ML candidates to make the system location-aware.

KARNATAKA_CROP_PRIORS = {
    # Northern Karnataka / irrigated and dryland systems
    "bagalkot": {
        "maize": 0.90, "pigeonpeas": 0.88, "chickpea": 0.82,
        "cotton": 0.76, "rice": 0.72, "groundnut": 0.68,
        "blackgram": 0.45, "mungbean": 0.35,
    },
    "belagavi": {
        "maize": 0.92, "rice": 0.86, "cotton": 0.82,
        "chickpea": 0.80, "pigeonpeas": 0.76, "groundnut": 0.62,
        "blackgram": 0.45,
    },
    "ballari": {
        "rice": 0.98, "cotton": 0.84, "maize": 0.78,
        "pigeonpeas": 0.72, "chickpea": 0.62, "groundnut": 0.52,
        "blackgram": 0.42,
    },
    "bidar": {
        "pigeonpeas": 0.92, "chickpea": 0.86, "maize": 0.78,
        "cotton": 0.72, "blackgram": 0.58, "groundnut": 0.55,
        "mungbean": 0.42,
    },
    "vijayapura": {
        "pigeonpeas": 0.92, "chickpea": 0.90, "cotton": 0.82,
        "maize": 0.74, "groundnut": 0.68, "blackgram": 0.48,
        "mungbean": 0.42,
    },
    "kalaburagi": {
        "pigeonpeas": 0.94, "chickpea": 0.88, "cotton": 0.82,
        "maize": 0.76, "blackgram": 0.58, "groundnut": 0.55,
    },
    "yadgir": {
        "pigeonpeas": 0.92, "chickpea": 0.84, "cotton": 0.82,
        "maize": 0.76, "groundnut": 0.56, "blackgram": 0.52,
    },
    "raichur": {
        "rice": 0.96, "cotton": 0.82, "pigeonpeas": 0.76,
        "maize": 0.72, "chickpea": 0.60, "groundnut": 0.54,
    },
    "koppal": {
        "rice": 0.98, "maize": 0.80, "cotton": 0.76,
        "pigeonpeas": 0.72, "chickpea": 0.62, "groundnut": 0.54,
    },
    "gadag": {
        "pigeonpeas": 0.92, "chickpea": 0.88, "cotton": 0.80,
        "maize": 0.72, "groundnut": 0.65, "blackgram": 0.48,
    },
    "dharwad": {
        "maize": 0.88, "cotton": 0.78, "pigeonpeas": 0.76,
        "chickpea": 0.74, "rice": 0.55, "groundnut": 0.58,
    },
    "haveri": {
        "maize": 0.88, "cotton": 0.80, "pigeonpeas": 0.76,
        "rice": 0.72, "chickpea": 0.68, "groundnut": 0.56,
    },
    "davanagere": {
        "maize": 0.92, "rice": 0.78, "cotton": 0.76,
        "chickpea": 0.68, "pigeonpeas": 0.66, "groundnut": 0.55,
    },
    "chitradurga": {
        "groundnut": 0.86, "pigeonpeas": 0.82, "maize": 0.74,
        "chickpea": 0.72, "cotton": 0.68, "blackgram": 0.48,
    },
    "tumakuru": {
        "groundnut": 0.88, "maize": 0.82, "pigeonpeas": 0.76,
        "chickpea": 0.72, "cotton": 0.62, "blackgram": 0.48,
    },
    "kolar": {
        "groundnut": 0.84, "maize": 0.78, "pigeonpeas": 0.72,
        "chickpea": 0.68, "banana": 0.50, "mango": 0.48,
    },
    "chikkaballapur": {
        "groundnut": 0.84, "maize": 0.80, "pigeonpeas": 0.72,
        "chickpea": 0.68, "blackgram": 0.48, "mango": 0.45,
    },
    "bengaluru rural": {
        "maize": 0.86, "groundnut": 0.80, "pigeonpeas": 0.72,
        "chickpea": 0.68, "banana": 0.50, "blackgram": 0.45,
    },
    "bengaluru urban": {
        "maize": 0.78, "groundnut": 0.70, "pigeonpeas": 0.66,
        "chickpea": 0.62, "banana": 0.48,
    },
    "ramanagara": {
        "groundnut": 0.80, "maize": 0.76, "pigeonpeas": 0.70,
        "chickpea": 0.62, "banana": 0.52, "mango": 0.48,
    },
    "mandya": {
        "rice": 0.92, "maize": 0.78, "pigeonpeas": 0.66,
        "chickpea": 0.60, "banana": 0.58, "groundnut": 0.55,
    },
    "mysuru": {
        "rice": 0.86, "maize": 0.82, "pigeonpeas": 0.68,
        "chickpea": 0.62, "banana": 0.58, "groundnut": 0.55,
    },
    "hassan": {
        "coffee": 0.88, "rice": 0.72, "maize": 0.70,
        "pigeonpeas": 0.58, "banana": 0.54, "chickpea": 0.50,
    },
    "chikkamagaluru": {
        "coffee": 0.98, "rice": 0.60, "maize": 0.58,
        "banana": 0.52, "pigeonpeas": 0.45,
    },
    "kodagu": {
        "coffee": 1.00, "banana": 0.58, "rice": 0.52,
        "coconut": 0.42, "mango": 0.35,
    },
    "shivamogga": {
        "rice": 0.88, "maize": 0.72, "coffee": 0.66,
        "banana": 0.58, "pigeonpeas": 0.46,
    },
    "dakshina kannada": {
        "coconut": 0.96, "banana": 0.78, "rice": 0.72,
        "coffee": 0.58, "papaya": 0.38,
    },
    "udupi": {
        "coconut": 0.98, "rice": 0.78, "banana": 0.72,
        "coffee": 0.52, "papaya": 0.35,
    },
    "uttara kannada": {
        "rice": 0.78, "coconut": 0.74, "banana": 0.62,
        "coffee": 0.55, "maize": 0.45,
    },
    "chamarajanagar": {
        "maize": 0.82, "rice": 0.72, "groundnut": 0.70,
        "pigeonpeas": 0.64, "banana": 0.54, "chickpea": 0.50,
    },
}

# Specific taluk evidence. These are local priors, not hard-coded final
# predictions. The strongest example is Gangavathi, where paddy is a major
# crop and therefore rice should not disappear merely because a generic ML
# dataset prefers another crop under the current weather snapshot.
TALUK_CROP_PRIORS = {
    "gangavathi": {
        "rice": 1.00,
        "maize": 0.62,
        "cotton": 0.58,
        "pigeonpeas": 0.48,
        "chickpea": 0.38,
    },
    "gangawati": {
        "rice": 1.00,
        "maize": 0.62,
        "cotton": 0.58,
        "pigeonpeas": 0.48,
        "chickpea": 0.38,
    },
    "lingasugur": {
        "rice": 0.92,
        "cotton": 0.78,
        "pigeonpeas": 0.68,
        "maize": 0.65,
        "chickpea": 0.52,
    },
    "lingsugur": {
        "rice": 0.92,
        "cotton": 0.78,
        "pigeonpeas": 0.68,
        "maize": 0.65,
        "chickpea": 0.52,
    },
    "sindhanur": {
        "rice": 1.00,
        "cotton": 0.76,
        "pigeonpeas": 0.62,
        "maize": 0.58,
    },
    "manvi": {
        "rice": 0.92,
        "cotton": 0.78,
        "pigeonpeas": 0.66,
        "maize": 0.62,
    },
    "devadurga": {
        "rice": 0.82,
        "cotton": 0.80,
        "pigeonpeas": 0.70,
        "maize": 0.62,
    },
    "badami": {
        "rice": 0.76,
        "maize": 0.82,
        "pigeonpeas": 0.74,
        "chickpea": 0.68,
        "cotton": 0.62,
    },
    "hospet": {
        "rice": 0.82,
        "cotton": 0.74,
        "maize": 0.68,
        "pigeonpeas": 0.60,
        "chickpea": 0.50,
    },
    "bellary": {
        "rice": 0.82,
        "cotton": 0.76,
        "maize": 0.68,
        "pigeonpeas": 0.60,
    },
    "ballari": {
        "rice": 0.82,
        "cotton": 0.76,
        "maize": 0.68,
        "pigeonpeas": 0.60,
    },
    "madikeri": {
        "coffee": 1.00,
        "banana": 0.52,
        "rice": 0.45,
        "coconut": 0.38,
    },
    "virajpet": {
        "coffee": 1.00,
        "banana": 0.52,
        "rice": 0.45,
        "coconut": 0.38,
    },
    "somwarpet": {
        "coffee": 1.00,
        "banana": 0.50,
        "rice": 0.44,
        "coconut": 0.36,
    },
}

# Generic fallback candidates for districts where a local profile is not
# available. This prevents the model from collapsing to fruit crops simply
# because the current weather is warm/humid.
GENERIC_KARNATAKA_CROPS = {
    "rice": 0.82,
    "maize": 0.80,
    "pigeonpeas": 0.76,
    "chickpea": 0.72,
    "cotton": 0.70,
    "groundnut": 0.68,
    "blackgram": 0.54,
    "mungbean": 0.48,
    "banana": 0.30,
    "coffee": 0.24,
    "coconut": 0.20,
    "papaya": 0.12,
}


def get_local_crop_prior(district, taluk, model_classes):
    """
    Return a Karnataka-aware prior for EVERY crop class in the trained model.

    Important fix:
    - The Random Forest still considers all 22 dataset crops.
    - Missing local-prior entries receive a small neutral baseline instead
      of zero, so they are not accidentally removed from consideration.
    - Coffee is given a very low local baseline outside the main coffee
      districts, while the existing district/taluk coffee priors remain
      strong in Kodagu, Chikkamagaluru and Hassan.
    - Coconut receives the same treatment outside coastal/high-coconut areas.

    These are decision-support weights, not probabilities.
    """
    taluk_key = clean_name(taluk)
    district_key = canonical_district(district)

    raw = TALUK_CROP_PRIORS.get(taluk_key)

    if raw is None:
        raw = KARNATAKA_CROP_PRIORS.get(district_key)

    if raw is None:
        raw = GENERIC_KARNATAKA_CROPS

    model_keys = {
        str(crop).strip().lower(): str(crop)
        for crop in model_classes
    }

    # Start with a small baseline for ALL trained crops.
    # This keeps all 22 classes available to the recommendation engine.
    filtered = {
        str(crop): 0.05
        for crop in model_classes
    }

    # Crops that are strongly location-dependent get a lower neutral
    # baseline when the selected district does not have a local priority.
    for crop in model_classes:
        crop_key = str(crop).strip().lower()

        if crop_key == "coffee":
            filtered[str(crop)] = 0.01

        elif crop_key == "coconut":
            filtered[str(crop)] = 0.02

    # Overlay the existing district/taluk agricultural context.
    for crop, weight in raw.items():
        actual = model_keys.get(str(crop).strip().lower())
        if actual is not None:
            filtered[actual] = float(weight)

    # If the selected location is a known coffee region, keep coffee
    # strongly supported even if a spelling/alias caused the district
    # dictionary lookup to miss it.
    coffee_districts = {
        "kodagu",
        "chikkamagaluru",
        "hassan",
    }

    if district_key in coffee_districts:
        actual = model_keys.get("coffee")
        if actual is not None and filtered.get(actual, 0.0) < 0.80:
            filtered[actual] = 0.90

    return filtered


def normalize_scores(score_map):
    """Normalize positive scores so they sum to one."""
    clean = {
        str(k): max(0.0, float(v))
        for k, v in score_map.items()
    }
    total = sum(clean.values())

    if total <= 0:
        n = max(1, len(clean))
        return {k: 1.0 / n for k in clean}

    return {
        k: v / total
        for k, v in clean.items()
    }


def rank_hybrid_crop_scores(
    model_probabilities,
    local_prior,
):
    """
    Combine Random Forest evidence and local Karnataka crop context.

    30% = ML probability
    70% = local crop prior

    This is a suitability score, NOT a calibrated probability.
    """
    ml = normalize_scores(model_probabilities)
    local = normalize_scores(local_prior)

    all_crops = set(ml) | set(local)

    combined = {}
    for crop in all_crops:
        ml_score = ml.get(crop, 0.0)
        local_score = local.get(crop, 0.0)

        combined[crop] = (
            0.30 * ml_score
            + 0.70 * local_score
        )

    return combined, ml, local


@app.route(
    "/api/recommend-crop/<district>/<taluk>",
    methods=["GET"],
)
def recommend_crop(district, taluk):

    try:

        if crop_model is None:
            return jsonify(
                {
                    "status": "error",
                    "error": "AI model is not loaded.",
                    "details": model_error,
                }
            ), 500

        # ----------------------------------------------------
        # A. Soil pH
        # ----------------------------------------------------
        soil_record = find_soil_record(
            district,
            taluk,
        )

        if soil_record is None:
            return jsonify(
                {
                    "status": "error",
                    "error": "Soil information not found.",
                }
            ), 404

        soil_ph = safe_float(
            soil_record.get("soil_ph")
        )

        if soil_ph is None:
            return jsonify(
                {
                    "status": "error",
                    "error": "Invalid soil pH value.",
                }
            ), 500

        # ----------------------------------------------------
        # B. Location
        # ----------------------------------------------------
        location = get_location_for_area(
            district,
            taluk,
        )

        latitude = location["latitude"]
        longitude = location["longitude"]

        # ----------------------------------------------------
        # C. Current weather
        # ----------------------------------------------------
        weather_source = "Open-Meteo current weather"
        weather_warning = None

        try:
            weather = get_weather_data(
                latitude,
                longitude,
            )

            temperature = safe_float(
                weather.get("temperature")
            )
            humidity = safe_float(
                weather.get("humidity")
            )
            current_rainfall = safe_float(
                weather.get("rain"),
                0.0,
            )

            if temperature is None or humidity is None:
                raise ValueError(
                    "Current temperature/humidity unavailable."
                )

        except Exception as weather_exc:
            iot = load_iot_data()
            iot_temperature = safe_float(
                iot.get("temperature")
            )
            iot_humidity = safe_float(
                iot.get("humidity")
            )

            if (
                iot_temperature is not None
                and iot_humidity is not None
            ):
                temperature = iot_temperature
                humidity = iot_humidity
                current_rainfall = 0.0
                weather_source = "ESP8266 live DHT22 fallback"
                weather_warning = (
                    "Online weather API unavailable; "
                    "temperature/humidity came from ESP8266."
                )
            else:
                temperature = 25.0
                humidity = 60.0
                current_rainfall = 0.0
                weather_source = (
                    "Fallback values - weather API unavailable"
                )
                weather_warning = str(weather_exc)

        # ----------------------------------------------------
        # D. Seasonal weather
        # ----------------------------------------------------
        seasonal_warning = None
        seasonal_source = "Open-Meteo ECMWF seasonal forecast"

        try:
            seasonal = get_seasonal_data(
                latitude,
                longitude,
            )

            four_month_rainfall = safe_float(
                seasonal.get("four_month_rainfall")
            )

            if four_month_rainfall is None:
                raise ValueError(
                    "Seasonal rainfall unavailable."
                )

        except Exception as seasonal_exc:
            # Never present current precipitation as a four-month forecast.
            # It is used only as a clearly labelled temporary ML fallback.
            four_month_rainfall = max(
                0.0,
                float(current_rainfall),
            )
            seasonal_source = (
                "Current precipitation fallback for ML only - "
                "NOT a four-month seasonal forecast"
            )
            seasonal_warning = (
                "Seasonal forecast unavailable; current precipitation "
                "was used only as an ML fallback. Details: "
                f"{seasonal_exc}"
            )

        # ----------------------------------------------------
        # E. Random Forest prediction
        # ----------------------------------------------------
        model_input = build_model_input(
            rainfall=four_month_rainfall,
            humidity=humidity,
            temperature=temperature,
            soil_ph=soil_ph,
        )

        print()
        print("========== HYBRID CROP INPUT ==========")
        print("District:", district)
        print("Taluk:", taluk)
        print("Soil pH:", soil_ph)
        print("Temperature:", temperature)
        print("Humidity:", humidity)
        print("Seasonal rainfall feature:", four_month_rainfall)
        print("Rainfall source:", seasonal_source)
        print("Model input:", model_input.to_dict(orient="records")[0])
        print("=======================================")

        prediction = crop_model.predict(
            model_input
        )

        if len(prediction) == 0:
            raise ValueError(
                "Random Forest returned no prediction."
            )

        ml_prediction = str(prediction[0])

        # ----------------------------------------------------
        # F. Random Forest probabilities
        # ----------------------------------------------------
        model_classes = [
            str(c)
            for c in getattr(
                crop_model,
                "classes_",
                [],
            )
        ]

        if not model_classes:
            model_classes = [ml_prediction]

        raw_probabilities = {}

        if hasattr(crop_model, "predict_proba"):
            probabilities = crop_model.predict_proba(
                model_input
            )[0]

            for crop_name, probability in zip(
                model_classes,
                probabilities,
            ):
                raw_probabilities[crop_name] = float(
                    probability
                )
        else:
            raw_probabilities = {
                ml_prediction: 1.0
            }

        # ----------------------------------------------------
        # G. Karnataka local crop prior
        # ----------------------------------------------------
        local_prior = get_local_crop_prior(
            district,
            taluk,
            model_classes,
        )

        combined, normalized_ml, normalized_local = (
            rank_hybrid_crop_scores(
                raw_probabilities,
                local_prior,
            )
        )

        # ====================================================
        # RANK ALL CROPS AVAILABLE IN THE TRAINED DATASET
        # ====================================================
        #
        # The uploaded Crop_recommendation dataset contains 22
        # crop classes. The Random Forest was trained on those
        # classes, so every class in model_classes is a valid
        # recommendation candidate.
        #
        # Karnataka local crop information influences the hybrid
        # suitability score, but it must NOT remove ML classes
        # from consideration.
        candidate_crops = set(model_classes)

        candidate_scores = {
            crop: combined.get(crop, 0.0)
            for crop in candidate_crops
        }

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        if not ranked:
            ranked = sorted(
                combined.items(),
                key=lambda item: item[1],
                reverse=True,
            )

        recommended_crop = ranked[0][0]
        best_hybrid_score = ranked[0][1]

        # ----------------------------------------------------
        # H. Top 5 transparent results
        # ----------------------------------------------------
        top_crops = []

        for crop, score in ranked[:5]:
            top_crops.append(
                {
                    "crop": crop,
                    "suitability_score": round(
                        score * 100.0,
                        1,
                    ),
                    "ml_probability": round(
                        normalized_ml.get(crop, 0.0) * 100.0,
                        1,
                    ),
                    "local_priority": round(
                        normalized_local.get(crop, 0.0) * 100.0,
                        1,
                    ),
                }
            )

        # Keep the frontend's existing 'scores' field working.
        scores = {
            crop: round(
                score * 100.0,
                1,
            )
            for crop, score in ranked
        }

        # The frontend currently displays 'confidence'. Preserve that
        # field, but explicitly identify it as a hybrid suitability score.
        confidence = round(
            60.0 + (best_hybrid_score * 40.0),
            1,
        )

        # ----------------------------------------------------
        # I. Local crop context
        # ----------------------------------------------------
        local_crop_context = None
        local_crop_source = None

        crop_reference = find_crop_reference(
            district,
            taluk,
        )

        if crop_reference:
            local_crop_context = {
                "primary_crop": crop_reference.get(
                    "primary_crop",
                    "",
                ),
                "secondary_crops": crop_reference.get(
                    "secondary_crops",
                    "",
                ),
            }
            local_crop_source = crop_reference.get(
                "source",
                "",
            )

        # ----------------------------------------------------
        # J. Latest IoT
        # ----------------------------------------------------
        iot_data = load_iot_data()

        # ----------------------------------------------------
        # K. Response
        # ----------------------------------------------------
        return jsonify(
            {
                "status": "success",
                "state": STATE,
                "district": district,
                "taluk": taluk,
                "soil_ph": round(soil_ph, 2),
                "temperature": round(temperature, 1),
                "humidity": round(humidity, 1),
                "rainfall": round(current_rainfall, 1),
                "four_month_rainfall": round(
                    four_month_rainfall,
                    1,
                ),
                "recommended_crop": recommended_crop,
                "confidence": confidence,
                "confidence_type": (
                    "Hybrid suitability score; not a calibrated probability"
                ),
                "model_probability": round(
                    max(raw_probabilities.values()) * 100.0,
                    1,
                ) if raw_probabilities else None,
                "ml_prediction": ml_prediction,
                "top_crops": top_crops,
                "scores": scores,
                "local_crop_context": local_crop_context,
                "local_crop_source": local_crop_source,
                "local_crop_prior_used": local_prior,
                "iot": iot_data,
                "model": "Random Forest + Karnataka Crop Prior",
                "method": (
                    "30% Random Forest climate/soil evidence + "
                    "70% Karnataka local crop context; all trained crop classes ranked"
                ),
                "location": location.get("name"),
                "location_source": location.get("source"),
                "weather_source": weather_source,
                "weather_warning": weather_warning,
                "seasonal_source": seasonal_source,
                "seasonal_warning": seasonal_warning,
                "note": (
                    "The recommendation is decision-support output. "
                    "The seasonal forecast is probabilistic and the "
                    "hybrid suitability score is not a calibrated probability."
                ),
            }
        )

    except Exception as exc:
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("AI RECOMMENDATION ERROR")
        print(str(exc))
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        traceback.print_exc()
        print()

        return jsonify(
            {
                "status": "error",
                "error": "AI crop recommendation failed.",
                "details": str(exc),
            }
        ), 500


# ============================================================
# 22. IoT POST API
# ============================================================

@app.route(
    "/api/iot-data",
    methods=["POST"],
)
def receive_iot_data():

    try:

        data = request.get_json(
            silent=True
        )

        if not data:
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "No JSON data received."
                    ),
                }
            ), 400

        normalized = normalize_iot_data(
            data
        )

        save_iot_data(
            normalized
        )

        print()
        print(
            "========== IoT DATA RECEIVED =========="
        )
        print(
            "Soil Moisture:",
            normalized[
                "soil_moisture"
            ],
            "%",
        )
        print(
            "Temperature:",
            normalized[
                "temperature"
            ],
            "C",
        )
        print(
            "Humidity:",
            normalized[
                "humidity"
            ],
            "%",
        )
        print(
            "Rain:",
            normalized["rain"],
        )
        print(
            "Updated:",
            normalized[
                "updated_at"
            ],
        )
        print(
            "========================================"
        )
        print()

        return jsonify(
            {
                "status": "success",
                "message": (
                    "IoT data received"
                ),
                "data": normalized,
            }
        )

    except Exception as exc:

        print(
            "IoT POST error:",
            exc,
        )
        traceback.print_exc()

        return jsonify(
            {
                "status": "error",
                "message": str(exc),
            }
        ), 500


# ============================================================
# 23. IoT GET API
# ============================================================

@app.route(
    "/api/iot-data",
    methods=["GET"],
)
def get_iot_data():

    data = load_iot_data()

    return jsonify(
        {
            "status": "success",
            "data": data,
        }
    )


# ============================================================
# 24. IRRIGATION PLANNING
# ============================================================

def rain_is_detected(rain):
    """Recognize the ESP8266 rain labels safely."""
    if rain is None:
        return False

    text = str(
        rain
    ).strip().upper()

    if text in {
        "NO RAIN",
        "NO_RAIN",
        "NORAIN",
        "DRY",
        "0",
        "FALSE",
    }:
        return False

    return (
        "RAIN" in text
        or text in {
            "WET",
            "YES",
            "1",
            "TRUE",
        }
    )


@app.route(
    "/api/irrigation/<district>/<taluk>",
    methods=["GET"],
    strict_slashes=False,
)
def irrigation_plan(
    district,
    taluk,
):

    try:

        # ----------------------------------------------------
        # A. Soil reference
        # ----------------------------------------------------

        soil_record = find_soil_record(
            district,
            taluk,
        )

        soil_ph = (
            safe_float(
                soil_record[
                    "soil_ph"
                ]
            )
            if soil_record
            else None
        )

        # ----------------------------------------------------
        # B. Latest ESP8266 data
        # ----------------------------------------------------

        iot = load_iot_data()

        soil_moisture = safe_float(
            iot.get(
                "soil_moisture"
            )
        )

        temperature = safe_float(
            iot.get(
                "temperature"
            )
        )

        humidity = safe_float(
            iot.get(
                "humidity"
            )
        )

        rain = iot.get(
            "rain"
        )

        # ----------------------------------------------------
        # C. No sensor yet
        # ----------------------------------------------------

        if soil_moisture is None:

            return jsonify(
                {
                    "status": "waiting",
                    "district": district,
                    "taluk": taluk,
                    "soil_ph": soil_ph,
                    "message": (
                        "Waiting for live "
                        "ESP8266 soil-moisture data."
                    ),
                    "iot_connected": False,
                }
            )

        # ----------------------------------------------------
        # D. Decision thresholds
        # ----------------------------------------------------

        target_moisture = 60.0

        if soil_moisture >= 60.0:

            irrigation_status = (
                "IRRIGATION NOT REQUIRED"
            )

            irrigation_level = "LOW"

            recommendation = (
                "Soil moisture is currently "
                "adequate. Continue monitoring."
            )

        elif soil_moisture >= 40.0:

            irrigation_status = (
                "MONITOR SOIL"
            )

            irrigation_level = "MODERATE"

            recommendation = (
                "Soil moisture is decreasing. "
                "Prepare for irrigation if it "
                "continues to fall."
            )

        else:

            irrigation_status = (
                "IRRIGATION REQUIRED"
            )

            irrigation_level = "HIGH"

            recommendation = (
                "Soil moisture is low. "
                "Irrigation is recommended."
            )

        # ----------------------------------------------------
        # E. Rain adjustment
        # ----------------------------------------------------

        if rain_is_detected(rain):

            if soil_moisture < target_moisture:

                irrigation_status = (
                    "DELAY IRRIGATION - RAIN DETECTED"
                )

                irrigation_level = "LOW"

                recommendation = (
                    "Rain is detected. Delay "
                    "irrigation temporarily and "
                    "continue monitoring soil moisture."
                )

        # ----------------------------------------------------
        # F. Prototype water estimate
        #
        # Demonstration field:
        # 100 mÂ²
        #
        # Assumption:
        # 1% moisture deficit â‰ˆ 0.5 L/mÂ²
        #
        # This is a prototype estimate, NOT a universal
        # agronomic irrigation formula.
        # ----------------------------------------------------

        moisture_deficit = max(
            0.0,
            target_moisture
            - soil_moisture,
        )

        water_per_m2 = (
            moisture_deficit
            * 0.5
        )

        demo_area = 100.0

        estimated_water = (
            water_per_m2
            * demo_area
        )

        if rain_is_detected(rain):
            estimated_water = 0.0

        estimated_water = round(
            estimated_water,
            1,
        )

        return jsonify(
            {
                "status": "success",
                "district": district,
                "taluk": taluk,

                "soil_ph": soil_ph,

                "soil_moisture": round(
                    soil_moisture,
                    1,
                ),

                "temperature": temperature,

                "humidity": humidity,

                "rain": rain,

                "target_moisture": (
                    target_moisture
                ),

                "irrigation_level": (
                    irrigation_level
                ),

                "irrigation_status": (
                    irrigation_status
                ),

                "estimated_water_liters": (
                    estimated_water
                ),

                "demo_field_area_m2": (
                    demo_area
                ),

                "recommendation": (
                    recommendation
                ),

                "iot_connected": True,

                "note": (
                    "Prototype irrigation planning "
                    "estimate. Field-scale irrigation "
                    "should be calibrated using crop, "
                    "soil type, root-zone depth, field "
                    "capacity and local agronomic conditions."
                ),
            }
        )

    except Exception as exc:

        print(
            "IRRIGATION ERROR:",
            exc,
        )
        traceback.print_exc()

        return jsonify(
            {
                "status": "error",
                "message": str(exc),
            }
        ), 500


# ============================================================
# 25. IRRIGATION TEST API
# ============================================================

@app.route(
    "/api/irrigation-test",
    methods=["GET"],
    strict_slashes=False,
)
def irrigation_test():

    return jsonify(
        {
            "status": "success",
            "message": (
                "Irrigation API is registered and running"
            ),
            "endpoint": (
                "/api/irrigation/<district>/<taluk>"
            ),
        }
    )


# ============================================================
# 26. OPTIONAL DEBUG / DATA SUMMARY API
# ============================================================

@app.route(
    "/api/system-info",
    methods=["GET"],
)
def system_info():

    return jsonify(
        {
            "status": "success",
            "project": (
                "AI and IoT-Based Agricultural "
                "Decision Support System"
            ),
            "state": STATE,
            "soil_records": len(
                soil_rows
            ),
            "crop_reference_records": len(
                crop_reference_rows
            ),
            "ai_model_loaded": (
                crop_model is not None
            ),
            "model_file": model_path_used,
            "model_features": (
                model_feature_columns()
            ),
            "iot_data_file": IOT_DATA_FILE,
            "soil_csv": SOIL_CSV,
            "seasonal_api": (
                OPEN_METEO_SEASONAL
            ),
            "weather_api": (
                OPEN_METEO_FORECAST
            ),
        }
    )


# ============================================================
# 27. TEXT-TO-SPEECH API (gTTS)
# ============================================================

@app.route(
    "/api/tts",
    methods=["POST"],
)
def text_to_speech():
    """Generate English/Kannada MP3 speech using gTTS."""

    try:
        if gTTS is None:
            return jsonify({
                "status": "error",
                "error": (
                    "gTTS is not installed. Run: "
                    "pip install --upgrade gTTS"
                ),
            }), 500

        data = request.get_json(silent=True) or {}
        text = str(data.get("text", "")).strip()
        language = str(data.get("language", "en")).strip().lower()

        if not text:
            return jsonify({
                "status": "error",
                "error": "No text was provided for voice generation.",
            }), 400

        if language not in {"en", "kn"}:
            return jsonify({
                "status": "error",
                "error": "Supported voice languages are en and kn.",
            }), 400

        audio_buffer = BytesIO()

        gTTS(
            text=text,
            lang=language,
            slow=False,
            tld="co.in",
        ).write_to_fp(audio_buffer)

        audio_buffer.seek(0)

        response = send_file(
            audio_buffer,
            mimetype="audio/mpeg",
            as_attachment=False,
            download_name="agriculture_voice.mp3",
            max_age=0,
        )

        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        return response

    except Exception as exc:
        print("TTS ERROR:", repr(exc))
        return jsonify({
            "status": "error",
            "error": f"Voice generation failed: {exc}",
        }), 500


# ============================================================
# 28. 404 HANDLER
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify(
        {
            "status": "error",
            "error": "API route not found.",
            "path": request.path,
        }
    ), 404


# ============================================================
# 29. GLOBAL EXCEPTION HANDLER
# ============================================================

@app.errorhandler(500)
def internal_error(error):

    print(
        "Unhandled Flask error:",
        error,
    )
    traceback.print_exc()

    return jsonify(
        {
            "status": "error",
            "error": "Internal server error.",
            "details": str(error),
        }
    ), 500


# ============================================================
# 30. ROUTE DISPLAY
# ============================================================

def print_startup_summary():

    print()
    print(
        "=================================================="
    )
    print(
        " SMART AGRICULTURE AI + IoT BACKEND"
    )
    print(
        "=================================================="
    )
    print(
        "State:",
        STATE,
    )
    print(
        "Soil records:",
        len(soil_rows),
    )
    print(
        "Crop reference records:",
        len(crop_reference_rows),
    )
    print(
        "AI model:",
        (
            "LOADED"
            if crop_model is not None
            else "NOT LOADED"
        ),
    )
    print(
        "Model file:",
        model_path_used,
    )
    print(
        "IoT file:",
        IOT_DATA_FILE,
    )
    print()
    print(
        "Registered routes:"
    )

    for rule in app.url_map.iter_rules():
        print(
            f"  {sorted(rule.methods)} "
            f"{rule}"
        )

    print()
    print(
        "Dashboard:",
        "http://127.0.0.1:5000",
    )
    print(
        "LAN access:",
        "http://<YOUR-PC-IP>:5000",
    )
    print(
        "=================================================="
    )
    print()


# ============================================================
# 30. START SERVER
# ============================================================

if __name__ == "__main__":

    print_startup_summary()

    app.run(
        host="0.0.0.0",
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
    )

