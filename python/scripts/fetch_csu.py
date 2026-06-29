"""Download CSU SSM/I or SSMIS daily FCDR-GRID files for a satellite and period.

    python -m scripts.fetch_csu SAT YEAR [--month MM] [--step N] [--workers W] [--out DIR]

SAT is like F13, F15, or F16. The sensor (SSM/I versus SSMIS) and the filename
prefix are inferred from the satellite number: F08 to F15 are SSM/I (CSU_SSMI_...),
F16 to F18 are SSMIS (CSU_SSMIS_...). Downloads every Nth day (default 4) from the
NCEI CSU FCDR-GRID tree (open HTTPS, no auth), W files at a time, with per-file
retries and cache-resume. Generalizes fetch_csu_f16 across the SSM/I era for
multi-decade work.
"""

import calendar
import concurrent.futures
import os
import socket
import sys
import urllib.request

BASE = ("https://www.ncei.noaa.gov/data/ssmis-brightness-temperature-csu/access/"
        "FCDR-GRID")


def prefer_ipv4_dns():
    """Sort IPv4 ahead of IPv6 for every connection in this process.

    The NCEI host resolves to both IPv4 and IPv6, and on hosts whose IPv6 route to
    NCEI is dead, urllib (which has no Happy Eyeballs and tries addresses in resolver
    order) stalls on the IPv6 attempt until the socket timeout, three times per file,
    so a fetch makes no progress. curl works on the same host because it races the
    families. Sorting IPv4 first makes urllib try the working route first and fall
    back to IPv6 only if IPv4 fails. Installed from main() rather than at import so
    importing this module's helpers does not change DNS ordering as a side effect.
    Idempotent, so a repeated call does not wrap the wrapper.
    """
    if getattr(socket.getaddrinfo, "_ipv4_first", False):
        return
    orig = socket.getaddrinfo

    def ipv4_first(*args, **kwargs):
        return sorted(orig(*args, **kwargs),
                      key=lambda ai: 0 if ai[0] == socket.AF_INET else 1)

    ipv4_first._ipv4_first = True
    socket.getaddrinfo = ipv4_first


def sensor_for(sat):
    return "SSMI" if int(sat[1:]) <= 15 else "SSMIS"


def fname(sat, date):
    return f"CSU_{sensor_for(sat)}_FCDR-GRID_V02R00_{sat}_D{date}.nc"


def url(sat, date):
    return f"{BASE}/{date[:4]}/{fname(sat, date)}"


def _fetch(sat, date, out):
    path = os.path.join(out, fname(sat, date))
    if os.path.exists(path):
        return "cached"
    for _ in range(3):
        try:
            urllib.request.urlretrieve(url(sat, date), path)
            return "got"
        except Exception:  # noqa: BLE001 - retry, then report the gap
            if os.path.exists(path):
                os.remove(path)
    return "miss"


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    sat = sys.argv[1].upper()
    year = int(sys.argv[2])
    rest = sys.argv[3:]
    step, workers, month, out = 4, 6, None, None
    if "--month" in rest:
        month = int(rest[rest.index("--month") + 1])
    if "--step" in rest:
        step = int(rest[rest.index("--step") + 1])
    if "--workers" in rest:
        workers = int(rest[rest.index("--workers") + 1])
    if "--out" in rest:
        out = rest[rest.index("--out") + 1]
    if out is None:
        out = f"../data/{sat.lower()}_{year}" + (f"{month:02d}" if month else "")
    os.makedirs(out, exist_ok=True)
    socket.setdefaulttimeout(45)
    prefer_ipv4_dns()

    months = [month] if month else range(1, 13)
    got = cached = miss = total = 0
    # A fresh executor per month. NCEI throttles a single large request queue (a
    # full-year run stalls after a handful of files), but one month at a time
    # completes reliably, and the teardown between months lets the rate limit recover.
    for mon in months:
        ndays = calendar.monthrange(year, mon)[1]
        dates = [f"{year:04d}{mon:02d}{day:02d}"
                 for day in range(1, ndays + 1, step)]
        total += len(dates)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for status in ex.map(lambda dt: _fetch(sat, dt, out), dates):
                got += status == "got"
                cached += status == "cached"
                miss += status == "miss"
        print(f"  {year}-{mon:02d}: {got} got, {cached} cached, {miss} gaps so far",
              flush=True)
    print(f"{out}: {got} downloaded, {cached} cached, {miss} gaps "
          f"({total} target days)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
