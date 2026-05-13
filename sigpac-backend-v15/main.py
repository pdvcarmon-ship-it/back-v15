from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import httpx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json
import io
import os
import hashlib
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import logging
from scipy.ndimage import generic_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SIGPAC Sentinel API", version="9.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

SIGPAC_CONSULTA_URL  = "https://sigpac-hubcloud.es/servicioconsultassigpac/query"
COPERNICUS_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
COPERNICUS_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

# Sentinel Hub Process API (requiere token OAuth)
PROCESS_API_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
# WMS de Copernicus Data Space
WMS_URL = "https://sh.dataspace.copernicus.eu/ogc/wms"

COPERNICUS_USER = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASS = os.getenv("COPERNICUS_PASS", "")
_token_cache = {"token": None, "expires_at": 0}


def cache_key(prefix: str, **kwargs) -> str:
    key = json.dumps(kwargs, sort_keys=True)
    return hashlib.md5(f"{prefix}_{key}".encode()).hexdigest()


async def get_copernicus_token() -> str:
    now = time.time()
    if _token_cache["token"] and now < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    if not COPERNICUS_USER or not COPERNICUS_PASS:
        raise HTTPException(status_code=500, detail="Credenciales Copernicus no configuradas.")
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            COPERNICUS_TOKEN_URL,
            data={"grant_type": "password", "username": COPERNICUS_USER,
                  "password": COPERNICUS_PASS, "client_id": "cdse-public"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        _token_cache["token"] = data["access_token"]
        _token_cache["expires_at"] = now + data.get("expires_in", 3600)
        logger.info("Token Copernicus obtenido")
        return _token_cache["token"]


def geojson_to_mask(geojson: dict, width: int, height: int, bbox: list) -> Image.Image:
    """Crea máscara binaria: blanco=dentro parcela, negro=fuera."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    def to_px(lon, lat):
        x = int((lon - min_lon) / lon_range * width)
        y = int((max_lat - lat) / lat_range * height)
        return (max(0, min(width-1, x)), max(0, min(height-1, y)))

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        geom_type = geom.get("type", "")
        if geom_type == "Polygon":
            rings = geom.get("coordinates", [])
            if rings:
                pts = [to_px(c[0], c[1]) for c in rings[0]]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=255)
                for ring in rings[1:]:
                    pts_h = [to_px(c[0], c[1]) for c in ring]
                    if len(pts_h) >= 3:
                        draw.polygon(pts_h, fill=0)
        elif geom_type == "MultiPolygon":
            for polygon in geom.get("coordinates", []):
                if polygon:
                    pts = [to_px(c[0], c[1]) for c in polygon[0]]
                    if len(pts) >= 3:
                        draw.polygon(pts, fill=255)
                    for ring in polygon[1:]:
                        pts_h = [to_px(c[0], c[1]) for c in ring]
                        if len(pts_h) >= 3:
                            draw.polygon(pts_h, fill=0)
    return mask


def aplicar_mascara_jpeg(img_bytes: bytes, geojson: dict, bbox: str) -> bytes:
    """
    Aplica máscara de parcela a imagen JPEG.
    Fuera de parcela → gris oscuro semitransparente para que se distinga.
    Devuelve JPEG.
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bbox_floats = list(map(float, bbox.split(",")))
    mask = geojson_to_mask(geojson, img.width, img.height, bbox_floats)
    mask_arr = np.array(mask)

    img_arr = np.array(img, dtype=np.uint8)
    # Fuera de parcela: oscurecer al 20% para distinguir sin transparencia
    outside = mask_arr < 128
    img_arr[outside] = (img_arr[outside] * 0.15).astype(np.uint8)

    result = Image.fromarray(img_arr)
    buf = io.BytesIO()
    result.save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def stats_dentro_parcela(img_bytes: bytes, geojson: dict, bbox: str, indice: str) -> dict:
    """Calcula estadísticas SOLO con los píxeles dentro de la parcela."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    bbox_floats = list(map(float, bbox.split(",")))
    mask = geojson_to_mask(geojson, img.width, img.height, bbox_floats)
    mask_arr = np.array(mask) > 128

    arr = np.array(img, dtype=np.float32) / 255.0
    verde = arr[:, :, 1]
    pixeles = verde[mask_arr]

    if len(pixeles) == 0:
        return {"indice": indice, "min": 0, "max": 0, "mean": 0, "std": 0}

    return {
        "indice": indice,
        "min": float(pixeles.min()),
        "max": float(pixeles.max()),
        "mean": float(pixeles.mean()),
        "std": float(pixeles.std()),
        "pixeles_parcela": int(mask_arr.sum()),
    }



def bbox_to_float(bbox_str: str):
    return list(map(float, bbox_str.split(",")))


async def procesar_sentinel_evalscript(
    bbox: str,
    fecha: str,
    evalscript: str,
    token: str,
    width: int = 1024,
    height: int = 1024,
) -> Optional[bytes]:
    """
    Usa la Sentinel Hub Process API para obtener imágenes procesadas.
    Esta API SÍ funciona con el token OAuth de Copernicus Data Space.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_to_float(bbox)

    # Fecha inicio y fin (día completo)
    fecha_dt = datetime.strptime(fecha, "%Y-%m-%d")
    fecha_inicio = fecha_dt.strftime("%Y-%m-%dT00:00:00Z")
    fecha_fin = (fecha_dt + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")

    payload = {
        "input": {
            "bounds": {
                "bbox": [min_lon, min_lat, max_lon, max_lat],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-2-l2a",
                "dataFilter": {
                    "timeRange": {"from": fecha_inicio, "to": fecha_fin},
                    "maxCloudCoverage": 80,
                }
            }]
        },
        "output": {
            "width": width,
            "height": height,
            "responses": [{"identifier": "default", "format": {"type": "image/jpeg", "parameters": {"quality": 95}}}]
        },
        "evalscript": evalscript,
    }

    try:
        async with httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "image/jpeg",
            },
            timeout=120,
        ) as client:
            resp = await client.post(PROCESS_API_URL, json=payload)
            logger.info(f"Process API status: {resp.status_code}")
            if resp.status_code == 200:
                return resp.content
            else:
                logger.error(f"Process API error: {resp.text[:300]}")
    except Exception as e:
        logger.error(f"Error Process API: {e}")
    return None


# Evalscripts para cada índice
EVALSCRIPTS = {
    "RGB": """
//VERSION=3
function setup() {
  return { input: ["B04", "B03", "B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
}
""",
    "NDVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-10);
  if (ndvi < -0.5) return [0.05, 0.05, 0.05];
  else if (ndvi < 0) return [0.75, 0.75, 0.75];
  else if (ndvi < 0.1) return [0.86, 0.86, 0.86];
  else if (ndvi < 0.2) return [1, 1, 0.88];
  else if (ndvi < 0.3) return [0.86, 0.96, 0.72];
  else if (ndvi < 0.4) return [0.56, 0.82, 0.54];
  else if (ndvi < 0.5) return [0.27, 0.67, 0.36];
  else if (ndvi < 0.6) return [0.13, 0.52, 0.26];
  else if (ndvi < 0.7) return [0.05, 0.39, 0.16];
  else return [0.0, 0.27, 0.09];
}
""",
    "NDWI": """
//VERSION=3
function setup() {
  return { input: ["B03", "B08"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndwi = (sample.B03 - sample.B08) / (sample.B03 + sample.B08 + 1e-10);
  let val = (ndwi + 1) / 2;
  return [1 - val, 1 - val, val];
}
""",
    "EVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04", "B02"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let evi = 2.5 * (sample.B08 - sample.B04) / (sample.B08 + 6*sample.B04 - 7.5*sample.B02 + 1 + 1e-10);
  let val = Math.min(Math.max((evi + 1) / 2, 0), 1);
  return [1 - val, val, 1 - val];
}
""",
    "NDRE": """
//VERSION=3
function setup() {
  return { input: ["B08", "B05"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndre = (sample.B08 - sample.B05) / (sample.B08 + sample.B05 + 1e-10);
  let val = Math.min(Math.max((ndre + 1) / 2, 0), 1);
  return [1 - val, val, 0.5 - val * 0.5];
}
""",
    "SAVI": """
//VERSION=3
function setup() {
  return { input: ["B08", "B04"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let savi = 1.5 * (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 0.5 + 1e-10);
  let val = Math.min(Math.max((savi + 1) / 2, 0), 1);
  return [1 - val, val, 0.2];
}
""",
}

INDICES_INFO = {
    "NDVI": "Normalized Difference Vegetation Index",
    "NDWI": "Normalized Difference Water Index",
    "EVI":  "Enhanced Vegetation Index",
    "NDRE": "Normalized Difference Red Edge",
    "SAVI": "Soil-Adjusted Vegetation Index",
}


# ── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "9.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "copernicus_configured": bool(COPERNICUS_USER and COPERNICUS_PASS),
    }


@app.get("/sigpac/punto")
async def get_parcela_por_punto(lat: float = Query(...), lon: float = Query(...)):
    ck = cache_key("sigpac_punto", lat=round(lat, 6), lon=round(lon, 6))
    cache_file = CACHE_DIR / f"sigpac_{ck}.geojson"
    if cache_file.exists():
        return JSONResponse(content=json.loads(cache_file.read_text()))

    url = f"{SIGPAC_CONSULTA_URL}/recinfobypoint/4326/{lon}/{lat}.geojson"
    try:
        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        if not data.get("features"):
            raise HTTPException(status_code=404, detail="No se encontró parcela.")
        cache_file.write_text(json.dumps(data))
        return JSONResponse(content=data)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error SIGPAC: {str(e)}")


@app.get("/sentinel/buscar")
async def buscar_imagenes(
    bbox: str = Query(...),
    fecha_inicio: str = Query(...),
    fecha_fin: str = Query(...),
    max_nubosidad: float = Query(30.0),
):
    try:
        min_lon, min_lat, max_lon, max_lat = map(float, bbox.split(","))
    except ValueError:
        raise HTTPException(status_code=400, detail="bbox invalido")

    footprint = (
        f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},"
        f"{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))"
    )
    params = {
        "$filter": (
            f"Collection/Name eq 'SENTINEL-2' "
            f"and ContentDate/Start gt {fecha_inicio}T00:00:00.000Z "
            f"and ContentDate/Start lt {fecha_fin}T23:59:59.000Z "
            f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
            f"and att/OData.CSC.DoubleAttribute/Value le {max_nubosidad}) "
            f"and OData.CSC.Intersects(area=geography'SRID=4326;{footprint}')"
        ),
        "$orderby": "ContentDate/Start desc",
        "$top": "10",
        "$expand": "Attributes",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(COPERNICUS_SEARCH_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
        productos = []
        for item in data.get("value", []):
            cloud = next((a["Value"] for a in item.get("Attributes", []) if a["Name"] == "cloudCover"), None)
            productos.append({
                "id": item["Id"],
                "nombre": item["Name"],
                "fecha": item["ContentDate"]["Start"][:10],
                "nubosidad": round(cloud, 1) if cloud is not None else None,
                "size_mb": round(item.get("ContentLength", 0) / 1e6, 1),
            })
        return {"total": len(productos), "productos": productos}
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error Copernicus: {e}")


@app.get("/imagen/rgb")
async def imagen_rgb(
    bbox: str = Query(...),
    fecha: str = Query(..., description="YYYY-MM-DD"),
    geojson: Optional[str] = Query(None, description="GeoJSON parcela para recorte"),
):
    """Imagen color natural recortada por geometría de parcela."""
    ck = cache_key("rgb10", bbox=bbox, fecha=fecha, mask=bool(geojson))
    cache_png = CACHE_DIR / f"{ck}_rgb.jpg"

    if cache_png.exists():
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/jpeg")

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_rgb(cache_png)

    img_bytes = await procesar_sentinel_evalscript(bbox, fecha, EVALSCRIPTS["RGB"], token)

    if not img_bytes:
        logger.warning("Process API falló para RGB, usando demo")
        return _demo_rgb(cache_png)

    # Aplicar máscara de parcela si se proporciona GeoJSON
    if geojson:
        try:
            geojson_data = json.loads(geojson)
            img_bytes = aplicar_mascara_jpeg(img_bytes, geojson_data, bbox)
            logger.info("Máscara RGB aplicada correctamente")
        except Exception as e:
            logger.warning(f"Error aplicando máscara RGB: {e}")

    cache_png.write_bytes(img_bytes)
    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/jpeg")


@app.get("/indices/lista")
async def lista_indices():
    return {k: {"descripcion": v, "evalscript": True} for k, v in INDICES_INFO.items()}


@app.get("/indice/calcular")
async def calcular_indice(
    bbox: str = Query(...),
    fecha: str = Query(..., description="YYYY-MM-DD"),
    indice: str = Query(...),
    geojson: Optional[str] = Query(None, description="GeoJSON parcela para recorte"),
    formato: str = Query("png"),
):
    """Calcula índice usando Sentinel Hub Process API con evalscript."""
    indice = indice.upper()
    if indice not in EVALSCRIPTS:
        raise HTTPException(status_code=400, detail=f"Indice desconocido: {list(INDICES_INFO.keys())}")

    ck = cache_key("indice6", bbox=bbox, fecha=fecha, idx=indice)
    cache_png = CACHE_DIR / f"{ck}.jpg"
    cache_stats = CACHE_DIR / f"{ck}_stats.json"

    if cache_png.exists() and formato == "png":
        return StreamingResponse(io.BytesIO(cache_png.read_bytes()), media_type="image/png")
    if cache_stats.exists() and formato == "stats":
        return JSONResponse(content=json.loads(cache_stats.read_text()))

    try:
        token = await get_copernicus_token()
    except HTTPException:
        return _demo_indice_simple(indice, cache_png, cache_stats, formato)

    png_bytes = await procesar_sentinel_evalscript(bbox, fecha, EVALSCRIPTS[indice], token)

    if not png_bytes:
        return _demo_indice_simple(indice, cache_png, cache_stats, formato)

    # Aplicar máscara y calcular estadísticas solo dentro de parcela
    if geojson:
        try:
            geojson_data = json.loads(geojson)
            stats = stats_dentro_parcela(png_bytes, geojson_data, bbox, indice)
            png_bytes = aplicar_mascara_jpeg(png_bytes, geojson_data, bbox)
            logger.info(f"Máscara índice aplicada: {stats.get('pixeles_parcela')} px en parcela")
        except Exception as e:
            logger.warning(f"Error máscara índice: {e}")
            img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0
            verde = arr[:, :, 1]
            stats = {"indice": indice, "min": float(verde.min()), "max": float(verde.max()),
                     "mean": float(verde.mean()), "std": float(verde.std())}
    else:
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        verde = arr[:, :, 1]
        stats = {"indice": indice, "min": float(verde.min()), "max": float(verde.max()),
                 "mean": float(verde.mean()), "std": float(verde.std())}

    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/jpeg")


def _demo_rgb(cache_png: Path):
    np.random.seed(123)
    size = (256, 256)
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    base = 0.5 + 0.3 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.2)
    r = np.clip(base * 80 + np.random.normal(0, 5, size), 50, 130).astype(np.uint8)
    g = np.clip(base * 120 + np.random.normal(0, 5, size), 80, 180).astype(np.uint8)
    b = np.clip(base * 50 + np.random.normal(0, 5, size), 30, 90).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    img = Image.fromarray(rgb, mode='RGB')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/jpeg")


def _demo_indice_simple(indice, cache_png, cache_stats, formato):
    np.random.seed(42)
    size = (256, 256)
    x, y = np.meshgrid(np.linspace(0, 1, size[1]), np.linspace(0, 1, size[0]))
    base = 0.3 + 0.4 * np.exp(-((x - 0.5)**2 + (y - 0.5)**2) / 0.15)
    vals = np.clip(base + np.random.normal(0, 0.05, size), 0, 1).astype(np.float32)

    stats = {"indice": indice, "min": float(vals.min()), "max": float(vals.max()),
             "mean": float(vals.mean()), "std": float(vals.std()), "modo": "DEMO"}
    cache_stats.write_text(json.dumps(stats))

    if formato == "stats":
        return JSONResponse(content=stats)

    cmaps = {"NDVI": "RdYlGn", "NDWI": "Blues", "EVI": "YlGn", "NDRE": "RdYlGn", "SAVI": "YlGn"}
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    fig.patch.set_facecolor('#0a0f0d')
    ax.set_facecolor('#0a1a0d')
    im = ax.imshow(vals, cmap=cmaps.get(indice, "RdYlGn"), vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"{indice} - DEMO", color='#e2ffe8', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor='#0a0f0d')
    plt.close()
    buf.seek(0)
    png_bytes = buf.read()
    cache_png.write_bytes(png_bytes)
    return StreamingResponse(io.BytesIO(png_bytes), media_type="image/jpeg")




# Zonas NDVI para mapa de producción
ZONAS_NDVI = [
    {"zona": 1, "label": "Zona 1", "min": 0.90, "max": 1.00, "color": (0,  80,  0)},
    {"zona": 2, "label": "Zona 2", "min": 0.80, "max": 0.89, "color": (0,  120, 0)},
    {"zona": 3, "label": "Zona 3", "min": 0.70, "max": 0.79, "color": (34, 170, 34)},
    {"zona": 4, "label": "Zona 4", "min": 0.60, "max": 0.69, "color": (100,200, 50)},
    {"zona": 5, "label": "Zona 5", "min": 0.50, "max": 0.59, "color": (220,220, 0)},
    {"zona": 6, "label": "Zona 6", "min": 0.40, "max": 0.49, "color": (255,180, 0)},
    {"zona": 7, "label": "Zona 7", "min": 0.30, "max": 0.39, "color": (255,120, 0)},
    {"zona": 8, "label": "Zona 8", "min": 0.20, "max": 0.29, "color": (220, 60, 0)},
    {"zona": 9, "label": "Zona 9", "min": 0.10, "max": 0.19, "color": (200, 30, 30)},
    {"zona":10, "label": "Zona 10","min": 0.00, "max": 0.09, "color": (140,  0,  0)},
]

PIXEL_AREA_M2 = 100       # 10m x 10m = 100 m² por pixel Sentinel-2
M2_PER_HA    = 10000.0   # m² por hectárea


@app.get("/ndvi/zonas")
async def ndvi_zonas(
    bbox: str = Query(...),
    fecha: str = Query(...),
    geojson: Optional[str] = Query(None),
):
    """
    Devuelve imagen NDVI coloreada por zonas de producción
    y estadísticas de superficie por zona.
    """
    ck = cache_key("ndvi_zonas11", bbox=bbox, fecha=fecha, mask=bool(geojson))
    cache_img = CACHE_DIR / f"{ck}_zonas.jpg"
    cache_data = CACHE_DIR / f"{ck}_zonas.json"

    if cache_img.exists() and cache_data.exists():
        return JSONResponse(content={
            "imagen_url": f"/ndvi/zonas/imagen?ck={ck}",
            "zonas": json.loads(cache_data.read_text()),
        })

    try:
        token = await get_copernicus_token()
    except HTTPException:
        raise HTTPException(status_code=500, detail="Error obteniendo token Copernicus")

    # Evalscript que devuelve NDVI en escala de grises (0-255 = NDVI -1 a 1)
    # Evalscript que codifica NDVI en canal R y G como uint8 (0-255)
    # R = parte alta del valor (floor), G = decimales, permite reconstruir NDVI con precisión
    evalscript_ndvi_rgb = """
//VERSION=3
function setup() {
  return { input: ["B08", "B04"], output: { bands: 3 } };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04 + 1e-10);
  // Codificar NDVI [-1,1] en dos canales uint8 para mayor precisión
  let v = Math.min(Math.max((ndvi + 1.0) / 2.0, 0), 1);  // [0,1]
  let high = Math.floor(v * 255);
  let low  = Math.floor((v * 255 - high) * 255);
  // Canal B = 128 como marcador para distinguir pixeles validos de fondo
  return [high / 255.0, low / 255.0, 0.5];
}
"""

    raw_bytes = await procesar_sentinel_evalscript(
        bbox, fecha, evalscript_ndvi_rgb, token, width=1024, height=1024
    )

    if not raw_bytes:
        raise HTTPException(status_code=502, detail="No se pudo obtener imagen Sentinel")

    # Decodificar NDVI desde imagen RGB de 2 canales
    img_rgb = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    rgb_arr = np.array(img_rgb, dtype=np.float32) / 255.0

    # Reconstruir valor [0,1] desde canal R (parte entera) y G (decimales)
    v = rgb_arr[:, :, 0] + rgb_arr[:, :, 1] / 255.0
    # Convertir de [0,1] a NDVI [-1,1]
    ndvi_arr = v * 2.0 - 1.0

    # Marcar píxeles sin datos (canal B ~ 0, no es 0.5)
    valid_mask = rgb_arr[:, :, 2] > 0.25
    ndvi_arr[~valid_mask] = -999  # valor centinela fuera de rango

    # Aplicar máscara de parcela - CRÍTICO: mismo bbox que la imagen
    mask_arr = None
    geojson_data = None
    if geojson:
        try:
            geojson_data = json.loads(geojson)
            bbox_floats = list(map(float, bbox.split(",")))
            # La máscara debe tener exactamente las mismas dimensiones que la imagen NDVI
            mask = geojson_to_mask(geojson_data, img_rgb.width, img_rgb.height, bbox_floats)
            mask_arr = np.array(mask) > 128
            logger.info(f"Máscara creada: {mask_arr.sum()} píxeles dentro de parcela de {img_rgb.width}x{img_rgb.height}")
        except Exception as e:
            logger.warning(f"Error máscara zonas: {e}")

    # Calcular área real de cada píxel en m²
    # Aproximación: 1 grado lat ≈ 111320m, 1 grado lon ≈ 111320 * cos(lat) m
    bbox_floats_calc = list(map(float, bbox.split(",")))
    min_lon_c, min_lat_c, max_lon_c, max_lat_c = bbox_floats_calc
    lat_center = (min_lat_c + max_lat_c) / 2.0
    import math
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))
    bbox_width_m  = (max_lon_c - min_lon_c) * m_per_deg_lon
    bbox_height_m = (max_lat_c - min_lat_c) * m_per_deg_lat
    bbox_area_m2  = bbox_width_m * bbox_height_m
    img_h, img_w  = ndvi_arr.shape
    total_pixels  = img_w * img_h
    pixel_area_m2_real = bbox_area_m2 / total_pixels
    logger.info(f"bbox área: {bbox_area_m2:.1f}m², píxel real: {pixel_area_m2_real:.2f}m²")

    # Interpolar píxeles sin datos (-999) usando vecinos válidos
    sin_datos = ndvi_arr <= -999
    if sin_datos.any() and (~sin_datos).any():
        # Para cada píxel sin datos, calcular la media de vecinos válidos (ventana 5x5)
        def rellenar_con_vecinos(values):
            centro = values[len(values) // 2]
            if centro > -999:
                return centro
            validos = values[values > -999]
            return float(validos.mean()) if len(validos) > 0 else centro

        ndvi_rellenado = generic_filter(
            ndvi_arr,
            rellenar_con_vecinos,
            size=5,
            mode='nearest'
        )
        # Solo aplicar el relleno donde había datos inválidos
        ndvi_arr = np.where(sin_datos, ndvi_rellenado, ndvi_arr)
        # Actualizar máscara de sin_datos
        sin_datos = ndvi_arr <= -999
        logger.info(f"Píxeles interpolados: {sin_datos.sum()} restantes sin datos")

    # Crear imagen coloreada por zonas
    h, w = ndvi_arr.shape
    rgb_arr = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_arr[:] = [30, 30, 30]  # Fondo oscuro (fuera de parcela)

    zonas_stats = []

    for zona in ZONAS_NDVI:
        in_zone = (ndvi_arr >= zona["min"]) & (ndvi_arr <= zona["max"]) & (ndvi_arr > -999)
        if mask_arr is not None:
            in_zone = in_zone & mask_arr

        pixel_count = int(in_zone.sum())
        # Área real de cada píxel según bbox y tamaño de imagen
        superficie_m2 = pixel_count * pixel_area_m2_real
        superficie_ha = superficie_m2 / M2_PER_HA

        rgb_arr[in_zone] = zona["color"]

        zonas_stats.append({
            "zona": zona["zona"],
            "label": zona["label"],
            "ndvi_min": zona["min"],
            "ndvi_max": zona["max"],
            "color_hex": "#{:02x}{:02x}{:02x}".format(*zona["color"]),
            "pixeles": pixel_count,
            "superficie_m2": float(round(superficie_m2, 1)),
            "superficie_ha": float(round(superficie_ha, 4)),
        })

    # Fuera de parcela: negro
    if mask_arr is not None:
        rgb_arr[~mask_arr] = [15, 15, 15]
    # Píxeles sin datos dentro de parcela: gris oscuro
    sin_datos = (ndvi_arr <= -999)
    if mask_arr is not None:
        sin_datos = sin_datos & mask_arr
    rgb_arr[sin_datos] = [40, 40, 40]

    img_result = Image.fromarray(rgb_arr, mode="RGB")
    buf = io.BytesIO()
    img_result.save(buf, format="JPEG", quality=92)
    img_bytes = buf.getvalue()

    cache_img.write_bytes(img_bytes)
    cache_data.write_text(json.dumps(zonas_stats))

    return JSONResponse(content={
        "imagen_url": f"/ndvi/zonas/imagen?ck={ck}",
        "zonas": zonas_stats,
    })


@app.get("/ndvi/zonas/imagen")
async def ndvi_zonas_imagen(ck: str = Query(...)):
    """Sirve la imagen de zonas NDVI cacheada."""
    cache_img = CACHE_DIR / f"{ck}_zonas.jpg"
    if not cache_img.exists():
        raise HTTPException(status_code=404, detail="Imagen no encontrada, recalcula primero")
    return StreamingResponse(io.BytesIO(cache_img.read_bytes()), media_type="image/jpeg")


@app.post("/ndvi/produccion")
async def calcular_produccion(
    datos: dict,
):
    """
    Calcula kg esperados por zona y total de parcela.
    Body: { "zonas": [...], "kg_por_ha": { "1": 5000, "2": 4500, ... } }
    """
    zonas = datos.get("zonas", [])
    kg_por_ha = datos.get("kg_por_ha", {})

    resultado = []
    total_kg = 0.0
    total_ha = 0.0

    for zona in zonas:
        zona_id = str(zona["zona"])
        sup_ha = zona.get("superficie_ha", 0)
        kg_ha = float(kg_por_ha.get(zona_id, 0))
        kg_zona = sup_ha * kg_ha

        resultado.append({
            "zona": zona["zona"],
            "label": zona["label"],
            "ndvi_min": zona["ndvi_min"],
            "ndvi_max": zona["ndvi_max"],
            "color_hex": zona["color_hex"],
            "superficie_ha": round(sup_ha, 4),
            "kg_por_ha": kg_ha,
            "kg_estimados": round(kg_zona, 1),
        })
        total_kg += kg_zona
        total_ha += sup_ha

    return {
        "zonas": resultado,
        "total_ha": round(total_ha, 4),
        "total_kg": round(total_kg, 1),
        "total_toneladas": round(total_kg / 1000, 3),
    }

@app.get("/cache/info")
async def cache_info():
    files = list(CACHE_DIR.glob("*"))
    total_mb = sum(f.stat().st_size for f in files if f.is_file()) / 1e6
    return {"archivos": len(files), "total_mb": round(total_mb, 2)}


@app.delete("/cache/limpiar")
async def limpiar_cache(dias: int = Query(7)):
    cutoff = time.time() - dias * 86400
    eliminados = sum(1 for f in CACHE_DIR.glob("*") if f.is_file() and f.stat().st_mtime < cutoff and not f.unlink())
    return {"eliminados": eliminados}
