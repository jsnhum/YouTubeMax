"""
attribution.py
--------------
Reusable sidebar attribution and citation widget for Streamlit apps.

Usage in any app::

    from attribution import render_sidebar_attribution
    render_sidebar_attribution(app_name="My Tool", version="1.0", ...)
"""

import streamlit as st


def render_sidebar_attribution(
    app_name: str = "YouTube Research Tool",
    version: str = "1.0",
    year: str = "2025",
    author_apa: str = "Svensson, J.",
    author_full: str = "Jonas Svensson",
    institution: str = "Linnaeus University",
    url: str = "",
    email: str = "",
    orcid: str = "",
    license_name: str = "MIT",
) -> None:
    """
    Render an attribution and citation block at the bottom of the sidebar.

    Parameters
    ----------
    app_name : str
        Display name of the application.
    version : str
        Current version string (e.g. "1.0", "2.3.1").
    year : str
        Publication / release year for the citation.
    author_apa : str
        Author name in APA format, e.g. "Svensson, J."
    author_full : str
        Full author name for the "Developed by" line.
    institution : str
        Institutional affiliation.
    url : str
        Canonical URL for the software (e.g. GitHub repo). Omitted if empty.
    email : str
        Contact e-mail address. Omitted if empty.
    orcid : str
        ORCID iD (digits only, e.g. "0000-0002-1825-0097"). Omitted if empty.
    license_name : str
        Short license identifier, e.g. "MIT", "CC BY 4.0".
    """
    st.sidebar.divider()

    st.sidebar.caption(f"Developed by **{author_full}**, {institution}")

    with st.sidebar.expander("📖 How to Cite This Tool"):
        st.write("If you use this tool in academic work, please cite it as:")

        # Build APA 7 software reference
        citation = f"{author_apa} ({year}). {app_name} (Version {version}) [Computer software]. {institution}."
        if url:
            citation += f" {url}"

        st.code(citation, language=None)

        if orcid:
            st.write(f"ORCID: https://orcid.org/{orcid}")

    if email:
        st.sidebar.caption(
            f"Licensed under {license_name} · [Contact](<mailto:{email}>)"
        )
    else:
        st.sidebar.caption(f"Licensed under {license_name}")
