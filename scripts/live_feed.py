"""Live exchange feed -> stdout, for piping into the C++ streaming engine.

Connects to a public crypto WebSocket feed (no API key needed) and prints one
"timestamp,price" line per sampling interval, flushed immediately. Pipe it into
qtda_stream to run the topology detector on a live market:

    python scripts/live_feed.py | ./cpp/build/qtda_stream --csv - --w 30 --eps auto

Venues: coinbase (default, BTC-USD style products) or binance (BTCUSDT style).
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone

import websockets


async def stream(venue: str, product: str, interval: float, duration: float | None):
    if venue == "coinbase":
        url = "wss://ws-feed.exchange.coinbase.com"
        subscribe = json.dumps(
            {"type": "subscribe", "product_ids": [product], "channels": ["ticker"]})
        price_of = lambda m: float(m["price"]) if m.get("type") == "ticker" else None
    elif venue == "binance":
        url = f"wss://stream.binance.com:9443/ws/{product.lower()}@trade"
        subscribe = None
        price_of = lambda m: float(m["p"]) if "p" in m else None
    else:
        raise ValueError(f"unknown venue {venue!r}")

    start = time.monotonic()
    last_emit = -float("inf")
    ticks = emitted = 0
    while True:
        try:
            async with websockets.connect(url, ping_interval=20) as ws:
                if subscribe:
                    await ws.send(subscribe)
                print(f"[feed] connected to {venue} {product}", file=sys.stderr)
                async for raw in ws:
                    if duration and time.monotonic() - start > duration:
                        print(f"[feed] done: {ticks} ticks, {emitted} samples emitted",
                              file=sys.stderr)
                        return
                    price = price_of(json.loads(raw))
                    if price is None:
                        continue
                    ticks += 1
                    now = time.monotonic()
                    if now - last_emit >= interval:
                        last_emit = now
                        emitted += 1
                        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                        print(f"{ts},{price}", flush=True)
        except (websockets.ConnectionClosed, OSError) as e:
            if duration and time.monotonic() - start > duration:
                return
            print(f"[feed] disconnected ({e}); reconnecting in 2s", file=sys.stderr)
            await asyncio.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--venue", choices=["coinbase", "binance"], default="coinbase")
    ap.add_argument("--product", default="BTC-USD",
                    help="e.g. BTC-USD (coinbase) or BTCUSDT (binance)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="seconds between emitted samples (default 1.0)")
    ap.add_argument("--duration", type=float, default=None,
                    help="stop after this many seconds (default: run forever)")
    args = ap.parse_args()
    try:
        asyncio.run(stream(args.venue, args.product, args.interval, args.duration))
    except KeyboardInterrupt:
        print("[feed] stopped", file=sys.stderr)


if __name__ == "__main__":
    main()
