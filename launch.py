"""
MILIEU — launch script
Starts the Flask server + ngrok tunnel in one command.
Run with: python launch.py
"""
import threading, sys, os

# ── ngrok tunnel ──────────────────────────────────────────────────────────────
try:
    from pyngrok import ngrok, conf

    # If you have a free ngrok account, paste your authtoken here:
    NGROK_TOKEN = "3DIigkEYyjsHTlzgWBA91Y9GA08_4cAHuUu8ydShAb6Vnn5F3"

    if NGROK_TOKEN:
        ngrok.set_auth_token(NGROK_TOKEN)

    tunnel = ngrok.connect(5001, "http")
    print("\n" + "="*55)
    print("  MILIEU is live at:")
    print(f"  {tunnel.public_url}")
    print("  Open that URL on your phone")
    print("="*55 + "\n")

except Exception as e:
    print(f"[ngrok] Could not start tunnel: {e}")
    print("[ngrok] Running locally only at http://localhost:5001\n")

# ── Flask server (imported last so ngrok URL prints first) ────────────────────
os.chdir(os.path.dirname(os.path.abspath(__file__)))
exec(open("server.py").read())
