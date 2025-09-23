"""
Google Drive backup helper: uploads local DB backup and CSV export to Drive.

Secrets (Streamlit):
  [google]
  client_id = "..."
  client_secret = "..."
  redirect_uri = "https://sah...streamlit.app"  # your app URL
  drive_folder_id = "..."  # optional; if absent will create one named 'ExpenseTrackerBackups'

Note: Streamlit Cloud sessions are ephemeral; tokens are kept in session state only.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload


SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def _get_flow() -> Flow:
    g = st.secrets.get("google", {})
    client_config = {
        "web": {
            "client_id": g.get("client_id"),
            "project_id": g.get("project_id", "expense-tracker"),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": g.get("client_secret"),
            "redirect_uris": [g.get("redirect_uri")],
            "javascript_origins": [g.get("redirect_uri")],
        }
    }
    flow = Flow.from_client_config(client_config, scopes=SCOPES)
    flow.redirect_uri = g.get("redirect_uri")
    return flow


def ensure_oauth() -> Optional[Credentials]:
    if "drive_creds" in st.session_state and st.session_state["drive_creds"]:
        return st.session_state["drive_creds"]

    # Step 1: present auth link
    flow = _get_flow()
    auth_url, _ = flow.authorization_url(prompt="consent", include_granted_scopes="true", access_type="offline")
    st.info("Authorize Google Drive to enable backups:")
    st.link_button("Authorize Drive", auth_url)

    # Step 2: Handle redirect code pasted manually (Streamlit Cloud workaround)
    code = st.text_input("Paste the 'code' param from the redirected URL here:")
    if code:
        try:
            flow.fetch_token(code=code)
            creds = flow.credentials
            st.session_state["drive_creds"] = creds
            st.success("Drive authorized!")
            return creds
        except Exception as e:
            st.error(f"Drive auth failed: {e}")
            return None
    return None


def _get_service(creds: Credentials):
    return build("drive", "v3", credentials=creds)


def _ensure_folder(service, folder_name: str = "ExpenseTrackerBackups", folder_id: Optional[str] = None) -> str:
    if folder_id:
        return folder_id
    # Try to find existing
    resp = service.files().list(q="mimeType='application/vnd.google-apps.folder' and name='ExpenseTrackerBackups' and trashed=false", fields="files(id,name)").execute()
    files = resp.get("files", [])
    if files:
        return files[0]["id"]
    # Create
    meta = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder"}
    folder = service.files().create(body=meta, fields="id").execute()
    return folder["id"]


def upload_bytes(creds: Credentials, folder_id: Optional[str], filename: str, data: bytes, mime: str = "application/octet-stream") -> str:
    service = _get_service(creds)
    target_folder = _ensure_folder(service, folder_id=folder_id)
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime)
    metadata = {"name": filename, "parents": [target_folder]}
    file = service.files().create(body=metadata, media_body=media, fields="id, webViewLink").execute()
    return file.get("webViewLink")

