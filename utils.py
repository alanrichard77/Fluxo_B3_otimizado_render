from datetime import datetime

def to_plotly_json(fig):
    # garante serialização limpa
    return {"data": fig.to_plotly_json().get("data", []),
            "layout": fig.to_plotly_json().get("layout", {})}

def parse_date(s: str) -> datetime:
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    # fallback
    return datetime.fromisoformat(s)
