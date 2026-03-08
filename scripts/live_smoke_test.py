#!/usr/bin/env python3
"""Smoke test untuk validasi live server sebelum deploy."""

import argparse
import sys
import time
from urllib.error import URLError, HTTPError
from urllib.request import urlopen
import json


def check_http(url: str, expected_status: int = 200, timeout: float = 5.0):
    try:
        with urlopen(url, timeout=timeout) as response:
            status = response.status
            body = response.read().decode("utf-8", errors="ignore")
            return status == expected_status, status, body
    except (URLError, HTTPError) as exc:
        return False, None, str(exc)


def main():
    parser = argparse.ArgumentParser(description="Smoke test live server (Streamlit + API)")
    parser.add_argument("--streamlit-url", default="http://127.0.0.1:8503", help="URL Streamlit")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000", help="Base URL FastAPI")
    parser.add_argument(
        "--streamlit-only",
        action="store_true",
        help="Lewati cek API dan hanya validasi Streamlit.",
    )
    parser.add_argument("--retries", type=int, default=20, help="Jumlah retry")
    parser.add_argument("--interval", type=float, default=1.5, help="Jeda antar retry (detik)")
    args = parser.parse_args()

    print("🔍 Mulai smoke test live server...")

    streamlit_ok = False
    api_ok = False

    for attempt in range(1, args.retries + 1):
        s_ok, s_status, s_body = check_http(args.streamlit_url)
        a_ok, a_status, a_body = check_http(f"{args.api_url}/health")

        streamlit_ok = s_ok
        api_ok = False

        if a_ok:
            try:
                health_json = json.loads(a_body)
                api_ok = bool(health_json.get("status"))
            except json.JSONDecodeError:
                api_ok = False

        if streamlit_ok and (api_ok or args.streamlit_only):
            print(f"✅ Streamlit OK ({args.streamlit_url})")
            if api_ok:
                print(f"✅ API health OK ({args.api_url}/health)")
            else:
                print("⚠️ API dilewati karena mode --streamlit-only aktif.")
            print("🎉 Smoke test lulus. Server siap untuk uji live pra-deploy.")
            return 0

        print(
            f"⏳ Attempt {attempt}/{args.retries}: "
            f"Streamlit={'OK' if streamlit_ok else 'WAIT'} "
            f"API={'OK' if api_ok else 'WAIT'}"
        )
        time.sleep(args.interval)

    print("❌ Smoke test gagal. Cek log server terlebih dahulu.")
    print(f"   Streamlit check: status={s_status}, detail={str(s_body)[:120]}")
    print(f"   API check: status={a_status}, detail={str(a_body)[:120]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
