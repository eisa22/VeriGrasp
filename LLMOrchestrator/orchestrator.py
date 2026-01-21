"""
LLMOrchestrator/orchestrator.py
LLM-basierte Entscheidung für Paket-Greif-Reihenfolge.

Nutzt GPT-4o-mini mit Vision-Capabilities um aus 3D-Screenshots
das optimale erste Paket zum Greifen zu identifizieren.
"""
import os
import sys
import base64
import json
from openai import OpenAI

# Config importieren
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


SYSTEM_PROMPT = """You are an intelligent logistics decision system for robotic parcel picking.
Your task is to analyze 3D point cloud visualizations of parcels on a pallet and determine which parcel should be picked FIRST by a robotic gripper.

Decision Criteria (in order of priority):
1. ACCESSIBILITY: The parcel must be freely accessible from above - not blocked by other parcels
2. STABILITY: Picking this parcel should not destabilize the remaining stack
3. SAFETY: The parcel should be in a stable position, not tilted or at risk of falling
4. EFFICIENCY: Prefer parcels that are easier to grip (top layer, clear edges)

Each parcel is labeled with a unique ID number (1, 2, 3, etc.) shown as a red number on a white disc above the parcel.

You will receive 3 images showing the same scene from different viewpoints to help you assess the 3D arrangement.

IMPORTANT: You MUST respond with valid JSON only, no additional text:
<number> - Get the number (red) from the parcel surface in the white circle

{"parcel_id": <number>}
"""


def encode_image_to_base64(image_path):
    """Konvertiert ein Bild zu Base64-String."""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def analyze_scene_for_picking(screenshot_paths, verbose=True):
    """
    Analysiert die 3 Screenshots und bestimmt das optimale erste Paket.
    
    Args:
        screenshot_paths: Liste mit Pfaden zu den 3 Viewpoint-Screenshots
        verbose: Wenn True, detaillierte Ausgabe im Terminal
        
    Returns:
        Dictionary mit parcel_id und reasoning
    """
    if verbose:
        print("\n" + "="*60)
        print("  LLM ORCHESTRATOR - Paket-Auswahl")
        print("="*60)
    
    # OpenAI Client initialisieren
    api_key = OPENAI_API_KEY
    if not api_key or api_key == "sk-...":
        raise ValueError("OPENAI_API_KEY nicht gesetzt!\n"
                        "Bitte in config.py eintragen: OPENAI_API_KEY = 'sk-...'")
    
    client = OpenAI(api_key=api_key)
    
    # Bilder für API vorbereiten
    image_contents = []
    for i, path in enumerate(screenshot_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Screenshot nicht gefunden: {path}")
        
        base64_image = encode_image_to_base64(path)
        image_contents.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{base64_image}",
                "detail": "high"
            }
        })
        if verbose:
            print(f"  [IMG] Viewpoint {i+1}: {os.path.basename(path)}")
    
    # User-Message mit Bildern zusammenstellen
    user_content = [
        {"type": "text", "text": "Analyze these 3 viewpoints of the pallet and determine which parcel should be picked first. Respond with JSON only."}
    ] + image_contents
    
    if verbose:
        print("\n[LLM] Sende Anfrage an GPT-4o-mini...")
    
    # API Call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        max_tokens=300,
        temperature=0.1  # Niedrige Temperatur für konsistente Entscheidungen
    )
    
    # Response parsen
    raw_response = response.choices[0].message.content.strip()
    
    if verbose:
        print(f"[LLM] Raw Response: {raw_response}")
    
    # JSON extrahieren
    try:
        # Versuche direkt zu parsen
        result = json.loads(raw_response)
    except json.JSONDecodeError:
        # Falls Markdown-Codeblock, extrahiere JSON
        if "```json" in raw_response:
            json_str = raw_response.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_str)
        elif "```" in raw_response:
            json_str = raw_response.split("```")[1].split("```")[0].strip()
            result = json.loads(json_str)
        else:
            raise ValueError(f"Konnte JSON nicht parsen: {raw_response}")
    
    # Ergebnis ausgeben
    if verbose:
        print("\n" + "-"*60)
        print(f"  ✓ ENTSCHEIDUNG: Paket #{result.get('parcel_id', '?')}")
        print(f"  Begründung: {result.get('reasoning', 'Keine Begründung')}")
        print("-"*60 + "\n")
    
    return result


def run_orchestrator(screenshot_paths):
    """
    Hauptfunktion für die Pipeline-Integration.
    
    Args:
        screenshot_paths: Liste mit Pfaden zu den 3 Screenshots
        
    Returns:
        Dictionary mit Ergebnis der LLM-Analyse
    """
    return analyze_scene_for_picking(screenshot_paths, verbose=True)
