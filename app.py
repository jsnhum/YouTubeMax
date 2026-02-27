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
from typing import Callable, Optional

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

        # Store results and invalidate any stale metadata from a previous search
        st.session_state.search_df = df
        st.session_state.search_query = query.strip()
        st.session_state.metadata_df = None

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
    st.rerun()


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
    # Add new tabs here as the app grows (transcripts, comments, channel info…)
    tab_search, tab_metadata = st.tabs(["🔍 Search", "📥 Download Metadata"])

    with tab_search:
        render_search_tab()

    with tab_metadata:
        render_metadata_tab()


if __name__ == "__main__":
    main()
