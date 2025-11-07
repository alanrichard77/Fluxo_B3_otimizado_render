import time

_cache = {}

def cache_get(key):
    item = _cache.get(key)
    if not item:
        return None
    value, expire = item
    if expire and time.time() > expire:
        _cache.pop(key, None)
        return None
    return value

def cache_set(key, value, TTL=3600):
    expire = time.time() + TTL if TTL else None
    _cache[key] = (value, expire)

def cache_clear():
    _cache.clear()
