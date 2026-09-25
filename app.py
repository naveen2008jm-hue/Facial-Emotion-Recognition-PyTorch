"""
app.py
======
Root WSGI application entrypoint for Vercel and local environments.
"""

from api.index import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
