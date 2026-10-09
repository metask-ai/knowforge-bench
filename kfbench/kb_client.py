"""HTTP client for one KnowForge knowledge base exposed as a read-only resource package.

Every method maps 1:1 onto a `/api/v2/kb/{uuid}/...` endpoint (the same ones the product's MCP server wraps).
Access needs a free account: register at https://mind.metask-ai.com, copy the account's API key, and add the
benchmark package UUIDs to that account once (`python -m kfbench.run subscribe`).  Every call carries the key as
`Authorization: Bearer <key>`.  Responses are returned as the server's `data` object.
"""
from __future__ import annotations

import asyncio
import time
import urllib.parse
from typing import Any

import requests

_MIME = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif",
         "bmp": "image/bmp", "tif": "image/tiff", "tiff": "image/tiff", "pdf": "application/pdf"}


class KBError(RuntimeError):
    pass


_AUTH_HINT = ("register at https://mind.metask-ai.com, copy the API key shown under 'MCP resource packages' "
              "into [kb] api_key (or KFBENCH_KB_API_KEY), then run `python -m kfbench.run subscribe`")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _check_auth(r: requests.Response) -> None:
    if r.status_code == 426:
        raise KBError(f"HTTP 426: the knowledge-base API now needs an account API key; {_AUTH_HINT}")
    if r.status_code == 401:
        raise KBError(f"HTTP 401: API key invalid or revoked; {_AUTH_HINT}")
    if r.status_code == 403:
        raise KBError(f"HTTP 403: this package is not in your account's enabled list; run `python -m kfbench.run subscribe` "
                      f"(or re-enable it on the 'MCP resource packages' page): {r.text[:200]}")


def subscribe(base_url: str, api_key: str, package_uuids: list[str], timeout_s: float = 60.0) -> dict:
    """Add package UUIDs to the account behind api_key (idempotent: already-added ones are skipped)."""
    r = requests.post(f"{base_url.rstrip('/')}/api/v2/me/subscriptions/import", json={"items": list(package_uuids)},
                      headers=_auth_headers(api_key), timeout=timeout_s)
    _check_auth(r)
    if r.status_code >= 400:
        raise KBError(f"subscribe failed HTTP {r.status_code}: {r.text[:300]}")
    j = r.json()
    return (j.get("data") if isinstance(j, dict) and "data" in j else j) or {}


class KB:
    def __init__(self, base_url: str, package_uuid: str, timeout_s: float = 180.0, rate_sleep_s: float = 0.0,
                 api_key: str = ""):
        self.base = base_url.rstrip("/")
        self.uuid = package_uuid
        self.headers = _auth_headers(api_key)
        self.timeout = timeout_s
        self.sleep = rate_sleep_s
        self.calls = 0
        self.search_s = 0.0

    # -- transport --------------------------------------------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base}/api/v2/kb/{urllib.parse.quote(self.uuid, safe='')}/{path}"

    def _unwrap(self, r: requests.Response) -> Any:
        if r.status_code == 429:
            raise KBError("rate limited (429); slow down or set kb.rate_sleep_s")
        _check_auth(r)
        if r.status_code >= 400:
            raise KBError(f"HTTP {r.status_code} {r.url}: {r.text[:300]}")
        j = r.json()
        if isinstance(j, dict) and "success" in j:
            if not j.get("success"):
                raise KBError(str(j.get("error") or j))
            return j.get("data")
        return j

    def _get(self, path: str, params: dict | None = None) -> Any:
        if self.sleep:
            time.sleep(self.sleep)
        self.calls += 1
        return self._unwrap(requests.get(self._url(path), params=params or {}, headers=self.headers, timeout=self.timeout))

    def _post(self, path: str, body: dict) -> Any:
        if self.sleep:
            time.sleep(self.sleep)
        self.calls += 1
        return self._unwrap(requests.post(self._url(path), json=body, headers=self.headers, timeout=self.timeout))

    # -- endpoints (sync) -------------------------------------------------------------------------------------
    def card(self) -> dict:
        return self._get("card") or {}

    def overview(self) -> dict:
        return self._get("overview") or {}

    def sources(self, min_contribution: int = 0) -> dict:
        return self._get("sources", {"min_contribution": min_contribution}) or {}

    _root_names: dict | None = None

    def root_names(self) -> dict:
        """root_node_id -> document file name (cached).  Used to attribute image/table hits to their document."""
        if self._root_names is None:
            try:
                self._root_names = {str(s.get("root_node_id")): str(s.get("name") or "")
                                    for s in (self.sources().get("sources") or []) if s.get("root_node_id")}
            except Exception:  # noqa: BLE001
                self._root_names = {}
        return self._root_names

    def search(self, query: str, topk: int = 8, score_fields: bool = True) -> dict:
        t0 = time.time()
        d = self._post("search", {"query": query, "topk": topk, "score_fields": bool(score_fields)}) or {}
        self.search_s += time.time() - t0
        return d

    def fragment_context(self, fragment_id: str, window: int = 1) -> dict:
        return self._get("fragment-context", {"fragment_id": fragment_id, "window": window}) or {}

    def book_toc(self, book: str) -> dict:
        return self._get("book-toc", {"book": book}) or {}

    def list_tables(self) -> dict:
        return self._get("tables") or {}

    def query_table(self, item: str, table_id: str = "") -> dict:
        return self._get("query-table", {"item": item, "table_id": table_id}) or {}

    def get_table(self, table_id: str) -> dict:
        return self._get("table", {"table_id": table_id}) or {}

    def list_media(self, book: str = "", kind: str = "all", page: str = "", heading: str = "") -> dict:
        params: dict[str, Any] = {"book": book, "kind": kind}
        if page:
            params["page"] = page
        if heading:
            params["heading"] = heading
        return self._get("list-media", params) or {}

    def find_figure(self, label: str, book: str = "") -> dict:
        return self._get("find-figure", {"label": label, "book": book}) or {}

    def media_context(self, fragment_id: str) -> dict:
        return self._get("media-context", {"fragment_id": fragment_id}) or {}

    def download(self, fragment_id: str = "", root_node_id: str = "") -> tuple[bytes, str, str]:
        """Returns (bytes, mime, filename).  fragment_id of an image/table hit -> the rendered asset (png)."""
        if self.sleep:
            time.sleep(self.sleep)
        self.calls += 1
        params = {"fragment_id": fragment_id} if fragment_id else {"root_node_id": root_node_id}
        r = requests.get(self._url("download"), params=params, headers=self.headers, timeout=self.timeout)
        _check_auth(r)
        if r.status_code >= 400:
            raise KBError(f"download failed HTTP {r.status_code}: {r.text[:200]}")
        name = ""
        disp = r.headers.get("content-disposition", "")
        if "filename=" in disp:
            name = disp.split("filename=", 1)[1].strip().strip('"')
        ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        mime = r.headers.get("content-type", "").split(";")[0].strip() or _MIME.get(ext, "application/octet-stream")
        return r.content, mime, name

    # -- async wrappers -----------------------------------------------------------------------------------------
    async def a(self, fn, *args, **kw):
        return await asyncio.to_thread(fn, *args, **kw)
