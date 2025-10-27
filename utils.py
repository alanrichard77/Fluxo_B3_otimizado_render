import json
from datetime import datetime
from dateutil import parser as dtparser

def to_plotly_json(fig):
    return json.loads(fig.to_json())

def parse_date(s):
    if not s:
        return None
    try:
        return dtparser.parse(s).date()
    except:
        return None

