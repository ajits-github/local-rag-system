"""Mint a signed JWT for local manual testing of the authenticated API boundary.

Auth-boundary milestone. For the default `HS256` algorithm, reads the same
`JWT_HS256_SECRET` env var `config.jwt_signing_key()` reads, so a token
minted here verifies successfully against a locally running `POST /query`
with `security.auth.enabled: true`. RS256/ES256 signing needs a PRIVATE
key this script does not read from config (config only names a PUBLIC key
path, for verification) -- pass `--private-key-path` explicitly for those.

Usage:
    JWT_HS256_SECRET=dev-secret python scripts/issue_dev_token.py \
        --sub alice --tenant tenant_alpha --roles tenant_alpha_admin
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import jwt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402


def main() -> None:
    """CLI entrypoint: build claims from args, sign, and print the token."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub", required=True, help="Subject claim (opaque caller id).")
    parser.add_argument("--tenant", default=None, help="tenant_id claim.")
    parser.add_argument(
        "--roles", default="", help="Comma-separated roles claim, e.g. 'tenant_alpha_admin'."
    )
    parser.add_argument(
        "--exp-seconds", type=int, default=3600, help="Seconds until expiration, default 3600."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--private-key-path",
        default=None,
        help="Required to sign with RS256/ES256 -- config only names a public "
        "key path, used for verification, not signing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    jwt_config = config.security.auth.jwt

    if jwt_config.algorithm == "HS256":
        key: str | bytes = config.jwt_signing_key()
    else:
        if not args.private_key_path:
            raise SystemExit(
                f"algorithm is {jwt_config.algorithm}; pass --private-key-path "
                "to a PEM private key to sign with."
            )
        key = Path(args.private_key_path).read_bytes()

    now = int(time.time())
    claims: dict[str, object] = {
        "sub": args.sub,
        "tenant_id": args.tenant,
        "roles": [role for role in args.roles.split(",") if role],
        "iat": now,
        "exp": now + args.exp_seconds,
    }
    if jwt_config.issuer:
        claims["iss"] = jwt_config.issuer
    if jwt_config.audience:
        claims["aud"] = jwt_config.audience

    token = jwt.encode(claims, key, algorithm=jwt_config.algorithm)
    print(token)


if __name__ == "__main__":
    main()
