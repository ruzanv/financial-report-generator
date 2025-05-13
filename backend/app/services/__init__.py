import logging
from pythonjsonlogger import jsonlogger

# Configure root logger once per worker
handler = logging.StreamHandler()
handler.setFormatter(jsonlogger.JsonFormatter())
logging.basicConfig(handlers=[handler], level=logging.INFO)