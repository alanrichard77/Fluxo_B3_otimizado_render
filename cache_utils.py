import time
from typing import Any, Optional

_CACHE = {}

def cache_get(key: str) -> Optional[Any]:
    now = time.time()
    item = _CACHE.get(key)
    if not item:
        return None
    val, exp = item
    if exp is not None and now > exp:
        _CACHE.pop(key, None)
        return None
    return val

def cache_set(key: str, value: Any, ttl: int = 3600):
    exp = time.time() + ttl if ttl else None
    _CACHE[key] = (value, exp)
