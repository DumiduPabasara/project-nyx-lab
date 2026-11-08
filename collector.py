"""Project Nyx collection. Source address is supplied only via NYX_SOURCE_URL.

Logs metadata only; never logs URL, HTML, response bodies, or credentials.
"""
import os
import re
from datetime import datetime
from playwright.sync_api import sync_playwright
from storage import insert_draw

DRAW_RE = re.compile(r'\bDraw\s*No\.?\s*:\s*(\d{1,7})\b', re.I)
DATE_RE = re.compile(r'\bDate\s*:\s*((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+[A-Za-z]+\s+\d{1,2},?\s+\d{4})\b', re.I)
RESULT_RE = re.compile(r'(?<![A-Za-z0-9])([A-Z])((?:\s*[0-9]){6})(?!\s*[0-9])', re.I)


def parse_draw(text, expected):
    """Only accept a result near the matching draw number and date."""
    number = DRAW_RE.search(text)
    if not number or int(number.group(1)) != expected:
        return None
    # Restrict result search to immediately following date, before prize information.
    tail = text[number.end():number.end() + 1400]
    date_match = DATE_RE.search(tail)
    if not date_match:
        return None
    try:
        when = datetime.strptime(date_match.group(1).replace(',', ''), '%A %B %d %Y').date().isoformat()
    except ValueError:
        return None
    after_date = tail[date_match.end():]
    after_date = re.split(r'<<\s*VIEW ALL|CHECK WINNINGS|Prize Structure', after_date, maxsplit=1, flags=re.I)[0]
    result = RESULT_RE.search(after_date[:250])
    if result is None:
        return None
    return dict(draw_number=expected, draw_date=when, letter=result.group(1).upper(), digits=re.sub(r'\s+', '', result.group(2)))


def _diagnostics(text, expected, status):
    """Aggregate diagnostics only; never publish site text or actual URL."""
    n = DRAW_RE.search(text)
    return (f'HTTP {status}; body_chars={len(text)}; '
            f'draw_label={bool(n)}; draw_matches={bool(n and int(n.group(1)) == expected)}; '
            f'date_label={bool(DATE_RE.search(text))}; '
            f'result_token={bool(RESULT_RE.search(text))}')


def collect(start, limit=25):
    source = os.environ.get('NYX_SOURCE_URL', '').strip().rstrip('/')
    if not source.startswith('https://') or '?' in source or '#' in source:
        raise RuntimeError('NYX_SOURCE_URL must contain a private HTTPS base URL')
    checked = added = 0
    status = 'complete'
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=['--no-sandbox'])
        try:
            page = browser.new_page(locale='en-US')
            for draw in range(start, start + limit):
                checked += 1
                try:
                    response = page.goto(f'{source}/{draw}', wait_until='domcontentloaded', timeout=45000)
                    http = response.status if response else 0
                    if not response or http >= 400:
                        status = f'stopped at draw {draw}: HTTP {http}'
                        break
                    # Some pages hydrate asynchronously. Try after DOMContentLoaded and again after a short delay.
                    row = None
                    body = ''
                    for wait in (900, 2000, 3000):
                        page.wait_for_timeout(wait)
                        body = page.locator('body').inner_text(timeout=15000)
                        row = parse_draw(body, draw)
                        if row:
                            break
                    if row is None:
                        status = f'stopped at draw {draw}: unrecognized page ({_diagnostics(body,draw,http)})'
                        break
                    added += int(insert_draw(row))
                except Exception as exc:
                    # Do not print exception strings: browser exceptions may include source URLs.
                    status = f'stopped at draw {draw}: {type(exc).__name__}'
                    break
        finally:
            browser.close()
    return {'checked': checked, 'added': added, 'status': status}
