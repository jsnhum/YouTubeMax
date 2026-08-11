"""
YouTubemax – YouTube Research Tool
===================================
A Streamlit app for searching YouTube and downloading video metadata via the
YouTube Data API v3. Designed for academic researchers who need to collect and
analyse video metadata at scale.

Architecture
------------
- All API / data functions live at the top of the file. They are pure Python
  (no Streamlit calls) and can be unit-tested independently.
- UI rendering functions (render_sidebar, render_search_tab, etc.) follow below.
- State is managed via st.session_state so results persist across widget interactions.
- To add a new feature: write a render_*_tab() function and add a tab in main().
"""

import os
import re
import time
from typing import Callable, Optional

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
    _TRANSCRIPT_AVAILABLE = True
except ImportError:
    _TRANSCRIPT_AVAILABLE = False

import isodate
import pandas as pd
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from attribution import render_sidebar_attribution

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTubemax – Research Tool",
    page_icon="▶️",
    layout="wide",
    menu_items={
        "About": (
            "**YouTubemax** – YouTube Research Tool for Academic Researchers\n\n"
            "Built with Streamlit and the YouTube Data API v3."
        ),
    },
)

# ── Constants ─────────────────────────────────────────────────────────────────
_API_SERVICE = "youtube"
_API_VERSION = "v3"
MAX_RESULTS_PER_PAGE = 50   # Hard limit enforced by the YouTube Data API
MAX_IDS_PER_BATCH = 50      # videos().list() accepts up to 50 IDs per call


# ══════════════════════════════════════════════════════════════════════════════
# API / Data Layer
# ──────────────────────────────────────────────────────────────────────────────
# Functions here are pure Python with no Streamlit dependencies.
# Add new API functions (e.g. for transcripts, comments, channel info) here.
# ══════════════════════════════════════════════════════════════════════════════


def build_youtube_client(api_key: str):
    """
    Create and return an authenticated YouTube Data API v3 resource object.

    Parameters
    ----------
    api_key : str
        A valid YouTube Data API v3 key.

    Returns
    -------
    googleapiclient.discovery.Resource
        Authenticated API client ready to make calls.
    """
    return build(_API_SERVICE, _API_VERSION, developerKey=api_key)


def search_videos(
    api_key: str,
    query: str,
    total: int = 200,
    order: str = "relevance",
    language: Optional[str] = None,
    location: Optional[str] = None,
    location_radius: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Search YouTube for videos matching *query* and return a lightweight results
    DataFrame. Paginates automatically via nextPageToken until *total* results
    are collected or the API reports no further pages.

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.
    query : str
        Search query string (same syntax as the YouTube website).
    total : int
        Desired number of results (default 200). Actual count may be lower if
        YouTube has fewer matching videos than requested.
    order : str
        Sort order – one of "relevance", "date", "viewCount", "rating".
    language : str, optional
        ISO 639-1 relevance-language filter (e.g. "en", "ar", "sv").
    location : str, optional
        Geo-coordinate string "lat,lng" for a location-restricted search.
    location_radius : str, optional
        Radius around *location* (e.g. "50km", "100mi").
    progress_callback : callable, optional
        Called as ``progress_callback(collected, total)`` after each API page
        so the UI can update a progress indicator.

    Returns
    -------
    pd.DataFrame
        Columns: video_id, channel_title, title, description, url
    """
    youtube = build_youtube_client(api_key)
    rows: list[dict] = []
    page_token: Optional[str] = None

    while len(rows) < total:
        per_page = min(MAX_RESULTS_PER_PAGE, total - len(rows))

        # Build request kwargs – only include optional params when provided
        kwargs: dict = dict(
            part="snippet",
            q=query,
            type="video",
            maxResults=per_page,
            order=order,
        )
        if language:
            kwargs["relevanceLanguage"] = language
        if location:
            kwargs["location"] = location
        if location_radius:
            kwargs["locationRadius"] = location_radius
        if page_token:
            kwargs["pageToken"] = page_token

        response = youtube.search().list(**kwargs).execute()

        for item in response.get("items", []):
            vid = item["id"]["videoId"]
            s = item["snippet"]
            rows.append(
                {
                    "video_id": vid,
                    "channel_title": s.get("channelTitle", ""),
                    "title": s.get("title", ""),
                    "description": s.get("description", ""),
                    "url": f"https://www.youtube.com/watch?v={vid}",
                }
            )

        if progress_callback:
            progress_callback(len(rows), total)

        page_token = response.get("nextPageToken")
        if not page_token:
            break  # YouTube has no further pages for this query

    return pd.DataFrame(rows)


def get_channel_video_ids(
    api_key: str,
    channel_id: str,
    progress_callback: Optional[Callable[[int, Optional[int]], None]] = None,
) -> list[str]:
    """
    Fetch every video ID uploaded by a YouTube channel.

    Resolves the channel's "uploads" playlist, then paginates through it via
    ``playlistItems().list`` – roughly 1 quota unit per page of 50 videos,
    far cheaper than ``search().list`` (~100 units per page).

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.
    channel_id : str
        Channel ID (starts with "UC…").
    progress_callback : callable, optional
        Called as ``progress_callback(collected, total)`` after each page.
        ``total`` is the channel's public video count, or ``None`` if
        unavailable.

    Returns
    -------
    list[str]
        Video IDs in upload order (newest first).

    Raises
    ------
    ValueError
        If no channel is found for *channel_id*.
    """
    youtube = build_youtube_client(api_key)

    channel_response = (
        youtube.channels()
        .list(part="contentDetails,statistics", id=channel_id)
        .execute()
    )
    items = channel_response.get("items", [])
    if not items:
        raise ValueError(f"No channel found for ID '{channel_id}'.")

    uploads_playlist_id = items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
    total_videos = _to_int(items[0].get("statistics", {}).get("videoCount"))

    video_ids: list[str] = []
    page_token: Optional[str] = None

    while True:
        kwargs: dict = dict(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=MAX_RESULTS_PER_PAGE,
        )
        if page_token:
            kwargs["pageToken"] = page_token

        response = youtube.playlistItems().list(**kwargs).execute()

        for item in response.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])

        if progress_callback:
            progress_callback(len(video_ids), total_videos)

        page_token = response.get("nextPageToken")
        if not page_token:
            break

    return video_ids


@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_metadata_batch(api_key: str, batch_ids: tuple) -> list[dict]:
    """
    Fetch full metadata for up to 50 video IDs in a single API call.

    Decorated with ``@st.cache_data`` so repeated look-ups of the same batch
    are served from cache without consuming API quota. A tuple is used for
    *batch_ids* (rather than a list) because tuples are hashable.

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.
    batch_ids : tuple
        Tuple of up to 50 video IDs.

    Returns
    -------
    list[dict]
        One dict per video with keys: id, channel, date, time, title,
        description, duration_seconds, tags, views, likes, favourites, comments.
    """
    youtube = build_youtube_client(api_key)
    response = (
        youtube.videos()
        .list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch_ids),
        )
        .execute()
    )

    records: list[dict] = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})

        # Split ISO 8601 datetime "2021-03-15T14:00:00Z" into date + time parts
        published_at: str = snippet.get("publishedAt", "")
        date_part, time_part = "", ""
        if "T" in published_at:
            date_part, time_part = published_at.split("T", 1)
            time_part = time_part.rstrip("Z")

        tags: list = snippet.get("tags", [])

        records.append(
            {
                "id": item["id"],
                "channel": snippet.get("channelTitle", ""),
                "date": date_part,
                "time": time_part,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "duration_seconds": _parse_iso_duration(content.get("duration", "")),
                "tags": "|".join(tags) if tags else "",
                "views": _to_int(stats.get("viewCount")),
                "likes": _to_int(stats.get("likeCount")),
                "favourites": _to_int(stats.get("favoriteCount")),
                "comments": _to_int(stats.get("commentCount")),
            }
        )

    return records


def get_video_metadata(
    api_key: str,
    video_ids: list[str],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Fetch full metadata for an arbitrary list of YouTube video IDs.

    Batches requests in groups of ``MAX_IDS_PER_BATCH`` (50) to stay within the
    API's per-request limit and minimise quota consumption. Individual batches
    are cached via ``_fetch_metadata_batch``, so previously downloaded IDs
    return instantly from cache.

    Parameters
    ----------
    api_key : str
        YouTube Data API v3 key.
    video_ids : list[str]
        YouTube video IDs to look up. Duplicates are removed automatically.
    progress_callback : callable, optional
        Called as ``progress_callback(batches_done, total_batches)`` after each
        completed batch so the UI can update a progress bar.

    Returns
    -------
    pd.DataFrame
        Columns: id, channel, date, time, title, description,
                 duration_seconds, tags, views, likes, favourites, comments.
    """
    # Deduplicate while preserving insertion order
    seen: set[str] = set()
    unique_ids = [v for v in video_ids if not (v in seen or seen.add(v))]  # type: ignore[func-returns-value]

    batches = [
        tuple(unique_ids[i : i + MAX_IDS_PER_BATCH])
        for i in range(0, len(unique_ids), MAX_IDS_PER_BATCH)
    ]
    total_batches = len(batches)
    all_records: list[dict] = []

    for i, batch in enumerate(batches):
        records = _fetch_metadata_batch(api_key, batch)
        all_records.extend(records)
        if progress_callback:
            progress_callback(i + 1, total_batches)

    return pd.DataFrame(all_records)


def parse_video_ids(text: str) -> list[str]:
    """
    Extract YouTube video IDs from a free-form text string.

    Handles:
    - Bare 11-character IDs (e.g. ``dQw4w9WgXcQ``)
    - Full ``youtube.com/watch?v=...`` URLs
    - ``youtu.be/...`` short URLs
    - Newline, comma, or space-separated inputs (any mix thereof)

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    list[str]
        Deduplicated list of video IDs in the order they first appeared.
    """
    url_re = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
    id_re = re.compile(r"^[A-Za-z0-9_-]{11}$")

    # Extract IDs embedded in URLs first
    ids_from_urls = url_re.findall(text)

    # Strip URLs from text, then split remaining tokens
    cleaned = url_re.sub(" ", text)
    bare_ids = [t for t in re.split(r"[\s,]+", cleaned) if id_re.match(t)]

    # Merge and deduplicate, preserving first-seen order
    seen: set[str] = set()
    unique: list[str] = []
    for vid in ids_from_urls + bare_ids:
        if vid and vid not in seen:
            seen.add(vid)
            unique.append(vid)
    return unique


def parse_channel_id(text: str) -> str:
    """
    Extract a bare channel ID from user input.

    Accepts a bare ID (e.g. ``UCxxxxxxxxxxxxxxxxxxxxxx``) or a full
    ``youtube.com/channel/UCxxxx…`` URL. Anything else is returned trimmed
    as-is so the API call can surface a clear "channel not found" error
    (handles/custom URLs like ``@name`` aren't resolved to IDs here).

    Parameters
    ----------
    text : str
        Raw user input.

    Returns
    -------
    str
        The extracted (or passed-through) channel ID.
    """
    text = text.strip()
    match = re.search(r"channel/([A-Za-z0-9_-]+)", text)
    return match.group(1) if match else text


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Serialise a DataFrame to a UTF-8-with-BOM CSV byte string.

    The BOM (byte order mark) ensures Excel on Windows opens the file with
    the correct encoding instead of garbling non-ASCII characters.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bytes
    """
    return df.to_csv(index=False).encode("utf-8-sig")


# ── Transcript functions (uses youtube-transcript-api, no API quota) ─────────


def fetch_transcript(video_id: str, preferred_languages: list[str]) -> dict:
    """
    Fetch the transcript for a single YouTube video.

    Tries each language in *preferred_languages* in order. If none are
    available, falls back to the first transcript YouTube offers (any language).
    Never raises – errors are captured in the ``"error"`` key of the result.

    Parameters
    ----------
    video_id : str
        YouTube video ID.
    preferred_languages : list[str]
        ISO 639-1 codes in preference order, e.g. ``["en", "ar"]``.

    Returns
    -------
    dict
        Keys: video_id, language, language_code, is_generated,
              transcript_text, error.
        ``transcript_text`` and language fields are ``None`` on failure;
        ``error`` is ``None`` on success.
    """
    base: dict = {
        "video_id": video_id,
        "language": None,
        "language_code": None,
        "is_generated": None,
        "transcript_text": None,
        "error": None,
    }

    if not _TRANSCRIPT_AVAILABLE:
        return {**base, "error": "youtube-transcript-api not installed"}

    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)

        # Try preferred languages; fall back to whatever YouTube has
        try:
            t = transcript_list.find_transcript(preferred_languages)
        except NoTranscriptFound:
            available = list(transcript_list)
            if not available:
                return {**base, "error": "No transcripts available for this video"}
            t = available[0]

        segments = t.fetch()
        text = " ".join(seg.text for seg in segments)
        return {
            **base,
            "language": t.language,
            "language_code": t.language_code,
            "is_generated": t.is_generated,
            "transcript_text": text,
        }

    except TranscriptsDisabled:
        return {**base, "error": "Transcripts disabled for this video"}
    except Exception as exc:
        return {**base, "error": str(exc)}


def get_transcripts(
    video_ids: list[str],
    preferred_languages: Optional[list[str]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    delay: float = 0.3,
) -> pd.DataFrame:
    """
    Fetch transcripts for a list of YouTube video IDs.

    Processes videos one at a time with a short *delay* between requests to
    avoid hitting YouTube's rate limits. Failures are recorded in the ``error``
    column rather than raising exceptions.

    Parameters
    ----------
    video_ids : list[str]
        YouTube video IDs to fetch transcripts for.
    preferred_languages : list[str], optional
        ISO 639-1 codes in preference order (default: ``["en"]``).
    progress_callback : callable, optional
        Called as ``progress_callback(done, total)`` after each video.
    delay : float
        Seconds to wait between requests (default 0.3 s).

    Returns
    -------
    pd.DataFrame
        Columns: video_id, language, language_code, is_generated,
                 transcript_text, error.
    """
    if preferred_languages is None:
        preferred_languages = ["en"]

    records: list[dict] = []
    total = len(video_ids)

    for i, vid in enumerate(video_ids):
        records.append(fetch_transcript(vid, preferred_languages))
        if progress_callback:
            progress_callback(i + 1, total)
        if i < total - 1:
            time.sleep(delay)

    return pd.DataFrame(records)


def _parse_language_input(text: str) -> list[str]:
    """
    Parse a comma-separated string of ISO 639-1 language codes into a list.

    Parameters
    ----------
    text : str
        e.g. ``"en, ar, sv"``

    Returns
    -------
    list[str]
        e.g. ``["en", "ar", "sv"]``. Falls back to ``["en"]`` if input is blank.
    """
    codes = [c.strip().lower() for c in text.split(",") if c.strip()]
    return codes if codes else ["en"]


def _detect_id_column(df: pd.DataFrame) -> Optional[str]:
    """
    Heuristically find the video-ID column in an uploaded DataFrame.

    Checks column names (case-insensitive) against a list of common names.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    str or None
        Column name if found, otherwise ``None``.
    """
    known = {"video_id", "videoid", "video id", "id", "youtube_id", "url", "link"}
    for col in df.columns:
        if col.strip().lower() in known:
            return col
    return None


# ── Private helpers ───────────────────────────────────────────────────────────


def _parse_iso_duration(iso_str: str) -> int:
    """
    Parse an ISO 8601 duration string to total seconds.

    Parameters
    ----------
    iso_str : str
        e.g. ``"PT4M13S"`` → 253 seconds.

    Returns
    -------
    int
        Duration in whole seconds, or 0 if *iso_str* is empty or unparseable.
    """
    if not iso_str:
        return 0
    try:
        return int(isodate.parse_duration(iso_str).total_seconds())
    except Exception:
        return 0


def _to_int(value) -> Optional[int]:
    """Convert *value* to ``int``, returning ``None`` if conversion fails."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════════════════════


def _init_session_state() -> None:
    """
    Initialise session state keys with defaults on first run.

    Called once at app start-up; subsequent runs leave existing values intact
    so that results, API keys, and other state persist across widget interactions.
    """
    defaults: dict = {
        # API key – loaded from env var if available, otherwise blank
        "api_key": os.environ.get("YOUTUBE_API_KEY", ""),
        # Latest search results (pd.DataFrame) and the query that produced them
        "search_df": None,
        "search_query": "",
        # Latest metadata download results (pd.DataFrame)
        "metadata_df": None,
        # Transcripts fetched from the Metadata tab (tied to metadata_df)
        "transcript_df": None,
        # Transcripts fetched from the Transcripts tab (upload / paste)
        "upload_transcript_df": None,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# UI – Sidebar
# ══════════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Render the sidebar with API key input and usage guidance."""
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.markdown(
            """
            ### YouTube API Key
            You need a free **YouTube Data API v3** key to use this app.

            **How to get one:**
            1. Open the [Google Cloud Console](https://console.cloud.google.com/)
            2. Create or select a project
            3. Go to **APIs & Services → Library**, search for
               **YouTube Data API v3**, and enable it
            4. Go to **APIs & Services → Credentials** and click
               **+ Create Credentials → API key**
            5. Copy the key and paste it in the field below

            Your key is stored only in this browser session and is sent
            exclusively to the YouTube API.
            """
        )

        api_key_input = st.text_input(
            "API Key",
            value=st.session_state.api_key,
            type="password",
            placeholder="AIzaSy…",
            help="Never persisted to disk – lives only in session memory.",
        )

        # Update session state whenever the field changes
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input

        if st.session_state.api_key:
            st.success("API key is set ✓")
        else:
            st.warning("No API key – enter one above to get started.")

        st.divider()

        st.markdown(
            """
            **Quota reference**
            | Operation | Units |
            |---|---|
            | Search (per 50 results) | ~100 |
            | Metadata (per 50 videos) | ~1 |
            | Daily free quota | 10,000 |

            Monitor usage in the
            [Google Cloud Console](https://console.cloud.google.com/apis/dashboard).
            """
        )

        render_sidebar_attribution()


# ══════════════════════════════════════════════════════════════════════════════
# UI – Tab 1: Search
# ══════════════════════════════════════════════════════════════════════════════


def render_search_tab() -> None:
    """
    Render the Search tab: input form, search logic, and results table.

    Stores results in ``st.session_state.search_df`` so they are available
    in the Download Metadata tab without re-running the search.
    """
    st.header("Search YouTube")
    st.caption(
        "Search for videos by keyword and collect their IDs. "
        "Switch to **Download Metadata** to fetch full statistics and details."
    )

    # ── Search form ───────────────────────────────────────────────────────────
    query = st.text_input(
        "Search query",
        placeholder='e.g.  Islamic finance documentary  |  "قرآن كريم"  |  al-Azhar',
        help="Supports the same operators as the YouTube website: quotes, minus sign, etc.",
    )

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        total = st.number_input(
            "Total results to collect",
            min_value=10,
            max_value=10_000,
            value=200,
            step=50,
            help="Each 50 results costs roughly 100 API quota units.",
        )
    with col2:
        order = st.selectbox(
            "Sort order",
            options=["relevance", "date", "viewCount", "rating"],
            help=(
                "**relevance** – best match first\n"
                "**date** – newest first\n"
                "**viewCount** – most watched first\n"
                "**rating** – highest rated first"
            ),
        )
    with col3:
        language = st.text_input(
            "Language filter",
            placeholder="e.g. en, ar, sv",
            help=(
                "ISO 639-1 two-letter code. YouTube will bias results toward "
                "videos in this language. Leave blank for any language."
            ),
        )

    with st.expander("📍 Location filter (optional)"):
        loc1, loc2 = st.columns(2)
        with loc1:
            location = st.text_input(
                "Coordinates (lat, lng)",
                placeholder="e.g. 59.33, 18.07",
                help="Restricts results to videos tagged near this geographic point.",
            )
        with loc2:
            location_radius = st.text_input(
                "Radius",
                placeholder="e.g. 50km",
                help="Accepts km or mi (e.g. '100km', '30mi'). Required when coordinates are set.",
            )

    search_btn = st.button(
        "🔍 Search",
        type="primary",
        disabled=not query.strip(),
        help="Enter a search query to enable this button.",
    )

    # ── Execute search ────────────────────────────────────────────────────────
    if search_btn:
        if not st.session_state.api_key:
            st.error(
                "Please enter your YouTube API key in the sidebar before searching."
            )
            st.stop()

        progress_bar = st.progress(0, text="Starting search…")
        status_text = st.empty()

        def on_search_progress(collected: int, wanted: int) -> None:
            frac = min(collected / max(wanted, 1), 1.0)
            progress_bar.progress(
                frac, text=f"Collected {collected} of {wanted} results…"
            )

        try:
            df = search_videos(
                api_key=st.session_state.api_key,
                query=query.strip(),
                total=int(total),
                order=order,
                language=language.strip() or None,
                location=location.strip() or None,
                location_radius=location_radius.strip() or None,
                progress_callback=on_search_progress,
            )
        except HttpError as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"YouTube API error: {exc}")
            st.stop()
        except Exception as exc:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Unexpected error: {exc}")
            st.stop()

        progress_bar.progress(1.0, text="Search complete!")
        status_text.empty()

        # Store results and invalidate stale metadata/transcripts from a previous search
        st.session_state.search_df = df
        st.session_state.search_query = query.strip()
        st.session_state.metadata_df = None
        st.session_state.transcript_df = None

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.search_df is not None:
        df: pd.DataFrame = st.session_state.search_df
        q: str = st.session_state.search_query

        st.success(f"**{len(df)}** results for: *{q}*")

        st.dataframe(
            df,
            column_config={
                "video_id": st.column_config.TextColumn("Video ID", width="small"),
                "url": st.column_config.LinkColumn("Watch", display_text="▶ Open"),
                "channel_title": st.column_config.TextColumn("Channel"),
                "title": st.column_config.TextColumn("Title"),
                "description": st.column_config.TextColumn("Description"),
            },
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            label="⬇ Download search results as CSV",
            data=to_csv_bytes(df),
            file_name=f"youtube_search_{q[:40].replace(' ', '_')}.csv",
            mime="text/csv",
            key="dl_search_csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# UI – Tab 2: Download Metadata
# ══════════════════════════════════════════════════════════════════════════════


def render_metadata_tab() -> None:
    """
    Render the Download Metadata tab.

    Provides two sources for video IDs:
    1. The results of the most recent search (loaded from session state).
    2. A text area where the user can paste IDs or URLs manually.

    After a successful download, results are stored in
    ``st.session_state.metadata_df`` and displayed below the inputs.
    """
    st.header("Download Video Metadata")
    st.caption(
        "Fetch full metadata – views, likes, duration, tags, and more – for any "
        "set of YouTube videos. Requests are batched in groups of 50 to minimise "
        "API quota usage."
    )

    # ── Source A: from the last search ───────────────────────────────────────
    st.subheader("From search results")

    if st.session_state.search_df is not None:
        n = len(st.session_state.search_df)
        q = st.session_state.search_query
        st.info(
            f"**{n} videos** from your last search (*{q}*) are ready. "
            "Click below to fetch their full metadata."
        )
        if st.button(
            f"📥 Download metadata for all {n} videos",
            type="primary",
            key="btn_meta_search",
        ):
            ids = st.session_state.search_df["video_id"].tolist()
            _run_metadata_download(ids)
    else:
        st.info(
            "No search results yet. Run a search on the **Search** tab first, "
            "then come back here."
        )

    st.divider()

    # ── Source B: manual ID / URL input ──────────────────────────────────────
    st.subheader("From video IDs or URLs")
    st.caption(
        "Paste any mix of bare video IDs, ``youtube.com/watch?v=…`` URLs, "
        "or ``youtu.be/…`` short URLs. Separate entries with newlines, commas, "
        "or spaces."
    )

    manual_text = st.text_area(
        "Video IDs / URLs",
        height=160,
        placeholder=(
            "dQw4w9WgXcQ\n"
            "https://www.youtube.com/watch?v=abc123def45\n"
            "xyz789abc12, qrs456def78"
        ),
        label_visibility="collapsed",
        key="meta_manual_ids",
    )

    if st.button("📥 Download metadata for pasted IDs", key="btn_meta_manual"):
        if not manual_text.strip():
            st.warning("Please paste at least one video ID or URL.")
        else:
            ids = parse_video_ids(manual_text)
            if not ids:
                st.warning(
                    "No valid YouTube video IDs found in the pasted text. "
                    "IDs are 11 characters long (letters, digits, hyphens, underscores)."
                )
            else:
                st.info(f"Parsed **{len(ids)}** unique video IDs.")
                _run_metadata_download(ids)

    st.divider()

    # ── Source C: entire channel by channel ID ───────────────────────────────
    st.subheader("From a channel ID")
    st.caption(
        "Enter a channel ID to fetch metadata for **every video the channel has "
        "uploaded**. Channel IDs start with `UC…` and can be found on the "
        "channel's 'About' page, or pasted as a full `youtube.com/channel/UC…` URL."
    )

    channel_id_input = st.text_input(
        "Channel ID",
        placeholder="e.g. UC_x5XG1OV2P6uZZ5FSM9Ttw",
        key="meta_channel_id",
    )

    if st.button(
        "📥 Download metadata for entire channel",
        key="btn_meta_channel",
        disabled=not channel_id_input.strip(),
        help="Enter a channel ID to enable this button.",
    ):
        channel_id = parse_channel_id(channel_id_input)
        _run_channel_download(channel_id)

    # ── Display cached metadata results ──────────────────────────────────────
    if st.session_state.metadata_df is not None:
        _render_metadata_results(st.session_state.metadata_df)


def _run_metadata_download(video_ids: list[str]) -> None:
    """
    Fetch metadata for *video_ids* with a progress bar and store the result.

    On success, stores the DataFrame in ``st.session_state.metadata_df`` and
    triggers a rerun so the results table renders in the correct position below
    the input controls.

    Parameters
    ----------
    video_ids : list[str]
        YouTube video IDs to look up.
    """
    if not st.session_state.api_key:
        st.error("Please enter your YouTube API key in the sidebar first.")
        return

    n = len(video_ids)
    total_batches = (n + MAX_IDS_PER_BATCH - 1) // MAX_IDS_PER_BATCH

    progress_bar = st.progress(
        0,
        text=f"Downloading metadata for {n} video(s) in {total_batches} batch(es)…",
    )
    status_text = st.empty()

    def on_meta_progress(done: int, total: int) -> None:
        frac = min(done / max(total, 1), 1.0)
        progress_bar.progress(frac, text=f"Batch {done} of {total} complete…")
        videos_done = min(done * MAX_IDS_PER_BATCH, n)
        status_text.text(f"Processed ~{videos_done} of {n} videos.")

    try:
        df = get_video_metadata(
            api_key=st.session_state.api_key,
            video_ids=video_ids,
            progress_callback=on_meta_progress,
        )
    except HttpError as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"YouTube API error: {exc}")
        return
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Unexpected error: {exc}")
        return

    progress_bar.progress(1.0, text="Download complete!")
    status_text.empty()

    st.session_state.metadata_df = df
    st.session_state.transcript_df = None  # Invalidate transcripts from previous metadata
    st.rerun()


def _run_channel_download(channel_id: str) -> None:
    """
    Resolve *channel_id* to its full list of uploaded video IDs, then hand off
    to ``_run_metadata_download`` to fetch metadata for all of them.

    Parameters
    ----------
    channel_id : str
        YouTube channel ID (e.g. ``UCxxxxxxxxxxxxxxxxxxxxxx``).
    """
    if not st.session_state.api_key:
        st.error("Please enter your YouTube API key in the sidebar first.")
        return

    progress_bar = st.progress(0, text="Looking up channel…")
    status_text = st.empty()

    def on_channel_progress(collected: int, total: Optional[int]) -> None:
        if total:
            frac = min(collected / total, 1.0)
            progress_bar.progress(frac, text=f"Found {collected} of ~{total} videos…")
        else:
            status_text.text(f"Found {collected} videos so far…")

    try:
        video_ids = get_channel_video_ids(
            api_key=st.session_state.api_key,
            channel_id=channel_id,
            progress_callback=on_channel_progress,
        )
    except ValueError as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(str(exc))
        return
    except HttpError as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"YouTube API error: {exc}")
        return
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Unexpected error: {exc}")
        return

    progress_bar.empty()
    status_text.empty()

    if not video_ids:
        st.warning("This channel has no public videos.")
        return

    st.info(f"Found **{len(video_ids)}** videos on this channel. Fetching metadata…")
    _run_metadata_download(video_ids)


def _render_metadata_results(df: pd.DataFrame) -> None:
    """
    Display a metadata DataFrame with summary statistics and a CSV download button.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``get_video_metadata()``.
    """
    st.divider()
    st.subheader(f"Metadata – {len(df)} videos")

    # Optional summary statistics for numeric columns
    with st.expander("📊 Summary statistics"):
        numeric_cols = ["views", "likes", "comments", "favourites", "duration_seconds"]
        cols_present = [c for c in numeric_cols if c in df.columns]
        if cols_present:
            st.dataframe(
                df[cols_present].describe().round(0),
                use_container_width=True,
            )
        else:
            st.write("No numeric columns to summarise.")

    # Build display DataFrame: add a clickable URL column derived from the id
    display_df = df.copy()
    if "id" in display_df.columns:
        display_df.insert(
            1, "url", "https://www.youtube.com/watch?v=" + display_df["id"]
        )

    col_config: dict = {}
    if "url" in display_df.columns:
        col_config["url"] = st.column_config.LinkColumn("Watch", display_text="▶ Open")
    if "views" in display_df.columns:
        col_config["views"] = st.column_config.NumberColumn("Views", format="%d")
    if "likes" in display_df.columns:
        col_config["likes"] = st.column_config.NumberColumn("Likes", format="%d")
    if "favourites" in display_df.columns:
        col_config["favourites"] = st.column_config.NumberColumn(
            "Favourites", format="%d"
        )
    if "comments" in display_df.columns:
        col_config["comments"] = st.column_config.NumberColumn("Comments", format="%d")
    if "duration_seconds" in display_df.columns:
        col_config["duration_seconds"] = st.column_config.NumberColumn(
            "Duration (s)", format="%d"
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config=col_config,
    )

    st.download_button(
        label="⬇ Download metadata as CSV",
        data=to_csv_bytes(df),  # Export original df (without the computed url column)
        file_name="youtube_metadata.csv",
        mime="text/csv",
        key="dl_metadata_csv",
    )

    # ── Transcript download for these same videos ─────────────────────────────
    st.divider()
    st.subheader("📄 Download Transcripts")
    st.caption(
        "Fetch transcripts for the videos above. Uses no API quota – "
        "transcripts are pulled directly from YouTube's player."
    )

    lang_input_meta = st.text_input(
        "Preferred languages (comma-separated ISO 639-1 codes)",
        value="en",
        placeholder="e.g. en, ar, sv",
        help=(
            "Languages are tried in order. If none match, the first available "
            "transcript for that video is used instead."
        ),
        key="meta_transcript_langs",
    )

    if st.button("📄 Fetch transcripts for these videos", key="btn_transcript_meta"):
        video_ids = df["id"].tolist()
        langs = _parse_language_input(lang_input_meta)
        _run_transcript_download(video_ids, langs, state_key="transcript_df")

    if st.session_state.transcript_df is not None:
        _render_transcript_results(st.session_state.transcript_df, key_suffix="meta")


# ══════════════════════════════════════════════════════════════════════════════
# UI – Tab 3: Transcripts (upload / paste)
# ══════════════════════════════════════════════════════════════════════════════


def render_transcripts_tab() -> None:
    """
    Render the Transcripts tab.

    Provides two ways to supply video IDs:
    1. Upload a CSV file that contains a column of video IDs or YouTube URLs.
    2. Paste IDs or URLs directly into a text area.

    Results are stored in ``st.session_state.upload_transcript_df``.
    No YouTube Data API quota is consumed.
    """
    st.header("Download Transcripts")
    st.caption(
        "Fetch transcripts for any set of YouTube videos. "
        "No API key or quota is needed – transcripts come directly from YouTube's player."
    )

    if not _TRANSCRIPT_AVAILABLE:
        st.error(
            "The `youtube-transcript-api` package is not installed. "
            "Add it to requirements.txt and restart the app."
        )
        return

    # ── Shared language preference ────────────────────────────────────────────
    lang_input = st.text_input(
        "Preferred languages (comma-separated ISO 639-1 codes)",
        value="en",
        placeholder="e.g. en, ar, sv",
        help=(
            "Languages are tried in the order you list them. "
            "If none are available, the first transcript YouTube offers is used."
        ),
        key="upload_transcript_langs",
    )
    langs = _parse_language_input(lang_input)

    # ── Source A: CSV upload ──────────────────────────────────────────────────
    st.subheader("From uploaded CSV")
    st.caption(
        "Upload a CSV that contains a column of video IDs or YouTube URLs. "
        "The app will auto-detect the right column, or let you pick one."
    )

    uploaded_file = st.file_uploader(
        "CSV file with video IDs",
        type=["csv"],
        help="The file must contain a column with YouTube video IDs or full URLs.",
    )

    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            df_upload = None

        if df_upload is not None:
            st.caption(f"Loaded {len(df_upload)} rows, {len(df_upload.columns)} columns.")

            id_col = _detect_id_column(df_upload)
            if id_col is None:
                id_col = st.selectbox(
                    "Which column contains video IDs or URLs?",
                    options=df_upload.columns.tolist(),
                    help="Pick the column that holds YouTube video IDs or watch URLs.",
                )
            else:
                st.info(f"Auto-detected ID column: **{id_col}**")

            raw_text = "\n".join(df_upload[id_col].astype(str).tolist())
            ids_from_file = parse_video_ids(raw_text)

            if ids_from_file:
                st.caption(f"Parsed **{len(ids_from_file)}** unique video IDs.")
                if st.button(
                    f"📄 Fetch transcripts for {len(ids_from_file)} videos from file",
                    key="btn_transcript_upload",
                    type="primary",
                ):
                    _run_transcript_download(ids_from_file, langs, state_key="upload_transcript_df")
            else:
                st.warning(
                    f"No valid YouTube video IDs found in column **{id_col}**. "
                    "IDs are 11 characters (letters, digits, hyphens, underscores)."
                )

    st.divider()

    # ── Source B: paste IDs / URLs ────────────────────────────────────────────
    st.subheader("From pasted IDs or URLs")
    st.caption(
        "Paste any mix of bare video IDs, ``youtube.com/watch?v=…`` URLs, "
        "or ``youtu.be/…`` short URLs. Separate with newlines, commas, or spaces."
    )

    paste_text = st.text_area(
        "Video IDs / URLs",
        height=160,
        placeholder=(
            "dQw4w9WgXcQ\n"
            "https://www.youtube.com/watch?v=abc123def45\n"
            "xyz789abc12, qrs456def78"
        ),
        label_visibility="collapsed",
        key="transcript_paste_ids",
    )

    if st.button("📄 Fetch transcripts for pasted IDs", key="btn_transcript_paste"):
        if not paste_text.strip():
            st.warning("Please paste at least one video ID or URL.")
        else:
            ids_from_paste = parse_video_ids(paste_text)
            if not ids_from_paste:
                st.warning("No valid YouTube video IDs found in the pasted text.")
            else:
                st.info(f"Parsed **{len(ids_from_paste)}** unique video IDs.")
                _run_transcript_download(
                    ids_from_paste, langs, state_key="upload_transcript_df"
                )

    # ── Display results ───────────────────────────────────────────────────────
    if st.session_state.upload_transcript_df is not None:
        _render_transcript_results(
            st.session_state.upload_transcript_df, key_suffix="upload"
        )


def _run_transcript_download(
    video_ids: list[str],
    preferred_languages: list[str],
    state_key: str,
) -> None:
    """
    Fetch transcripts for *video_ids* with a per-video progress bar.

    Stores the resulting DataFrame in ``st.session_state[state_key]`` and
    triggers a rerun so results render below the controls.

    Parameters
    ----------
    video_ids : list[str]
        YouTube video IDs.
    preferred_languages : list[str]
        ISO 639-1 codes in preference order.
    state_key : str
        Session state key to store results in (``"transcript_df"`` or
        ``"upload_transcript_df"``).
    """
    n = len(video_ids)
    progress_bar = st.progress(0, text=f"Fetching transcripts for {n} video(s)…")
    status_text = st.empty()

    def on_progress(done: int, total: int) -> None:
        frac = min(done / max(total, 1), 1.0)
        progress_bar.progress(frac, text=f"Video {done} of {total}…")
        status_text.text(f"Fetched {done} transcript(s) so far.")

    try:
        df = get_transcripts(
            video_ids=video_ids,
            preferred_languages=preferred_languages,
            progress_callback=on_progress,
        )
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Unexpected error: {exc}")
        return

    progress_bar.progress(1.0, text="Done!")
    status_text.empty()

    st.session_state[state_key] = df
    st.rerun()


def _render_transcript_results(df: pd.DataFrame, key_suffix: str = "") -> None:
    """
    Display transcript results with a summary, preview table, and CSV download.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``get_transcripts()``.
    key_suffix : str
        Appended to widget keys to avoid collisions when this function is
        called from multiple tabs.
    """
    n_total = len(df)
    n_ok = int(df["transcript_text"].notna().sum())
    n_fail = n_total - n_ok

    st.divider()
    st.subheader(f"Transcripts – {n_ok} of {n_total} videos")

    if n_fail > 0:
        st.warning(
            f"**{n_fail}** video(s) had no transcript available "
            "(disabled, private, or no captions)."
        )
        with st.expander("Show unavailable videos"):
            failed_df = df[df["transcript_text"].isna()][["video_id", "error"]]
            st.dataframe(failed_df, use_container_width=True, hide_index=True)

    # Build display table: add link column and truncated preview
    display_df = df.copy()
    display_df.insert(
        1, "url", "https://www.youtube.com/watch?v=" + display_df["video_id"]
    )
    display_df["preview"] = (
        display_df["transcript_text"]
        .fillna("")
        .str[:300]
        .where(display_df["transcript_text"].notna(), other="—")
    )
    display_df = display_df.drop(columns=["transcript_text"])

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("Watch", display_text="▶ Open"),
            "video_id": st.column_config.TextColumn("Video ID"),
            "language": st.column_config.TextColumn("Language"),
            "language_code": st.column_config.TextColumn("Code"),
            "is_generated": st.column_config.CheckboxColumn("Auto-generated"),
            "preview": st.column_config.TextColumn("Transcript preview"),
            "error": st.column_config.TextColumn("Error"),
        },
    )

    st.download_button(
        label="⬇ Download full transcripts as CSV",
        data=to_csv_bytes(df),
        file_name="youtube_transcripts.csv",
        mime="text/csv",
        key=f"dl_transcripts_{key_suffix}",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    App entry point.

    Initialises session state, renders the sidebar, then builds the tabbed main
    area. To add a new feature:
    1. Write a ``render_*_tab()`` function following the pattern above.
    2. Add a new tab entry in the ``st.tabs()`` call below.
    """
    _init_session_state()
    render_sidebar()

    st.title("▶ YouTubemax – YouTube Research Tool")
    st.caption(
        "Search YouTube and download video metadata for academic research. "
        "All data lives in your browser session and is exported as CSV – "
        "nothing is stored on any server."
    )

    # ── Tabs ──────────────────────────────────────────────────────────────────
    # Add new tabs here as the app grows (comments, channel info…)
    tab_search, tab_metadata, tab_transcripts = st.tabs(
        ["🔍 Search", "📥 Download Metadata", "📄 Transcripts"]
    )

    with tab_search:
        render_search_tab()

    with tab_metadata:
        render_metadata_tab()

    with tab_transcripts:
        render_transcripts_tab()


if __name__ == "__main__":
    main()
