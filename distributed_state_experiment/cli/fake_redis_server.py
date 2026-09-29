"""Run a real, TCP-listening `fakeredis.TcpFakeServer` as a local Redis stand-in.

Why this exists
----------------
This experiment was built in a sandboxed session where `docker pull
redis:7-alpine` was coordinated with a sibling agent doing unrelated
Docker work on the same host and ultimately not run in this session (see
`README.md`'s "What was executed vs. simulated" section), and no
`redis-server` binary is installed locally. `fakeredis.TcpFakeServer`
implements the real Redis wire protocol (RESP) over a real TCP socket --
`redis-py`, `redis-cli`, and every script in this directory talk to it
exactly as they would a genuine `redis-server`, including from separate
OS processes -- so this is a faithful stand-in for the actual multi-
process experiments (separate worker processes, killing one, scaling
worker count) even though the server implementation itself is a
pure-Python simulator, not the genuine C `redis-server` binary.

Streams (`XADD`/`XREADGROUP`/`XACK`/`XCLAIM`/`XPENDING`), `SET ... NX
PX`, `ZADD`/`ZRANGEBYSCORE`, and Lua scripting (`EVAL`, via the `lupa`
extra) are all exercised for real against this server in this
experiment's runs -- nothing about the *commands* used is faked, only
the server process itself.

To run the same experiments against genuine Redis instead, bring up
`docker-compose.redis.yml` (`docker compose -f docker-compose.redis.yml
up -d`) and pass `--redis-url redis://localhost:6379/0` to every script
in this directory; nothing else changes.
"""

from __future__ import annotations

import argparse
import logging

from fakeredis import TcpFakeServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("fake_redis_server")


def main() -> None:
    """Parse args and run the fake Redis TCP server until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6399)
    args = parser.parse_args()

    server = TcpFakeServer((args.host, args.port))
    logger.info(
        "fakeredis TCP server listening on %s:%d (redis://%s:%d/0) -- Ctrl+C to stop",
        args.host,
        args.port,
        args.host,
        args.port,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("stopping")


if __name__ == "__main__":
    main()
