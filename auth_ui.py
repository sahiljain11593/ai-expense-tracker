"""
Minimal Google Sign-In (via Firebase Web SDK) embedded into Streamlit using components.html.

Setup (Streamlit Secrets):
  [firebase]
  apiKey = "..."
  authDomain = "..."
  projectId = "..."
  appId = "..."

  [auth]
  allowed_email = "your@gmail.com"  # single-user gating

Returns a dict with keys: email, displayName, uid when signed in, else None.
"""

from __future__ import annotations

import json
import streamlit as st
import streamlit.components.v1 as components


def google_sign_in_widget() -> dict | None:
    firebase = st.secrets.get("firebase", {})
    api_key = firebase.get("apiKey")
    auth_domain = firebase.get("authDomain")
    project_id = firebase.get("projectId")
    app_id = firebase.get("appId")

    if not (api_key and auth_domain and project_id and app_id):
        st.warning("Firebase config missing in Streamlit secrets. Set [firebase] keys to enable Google login.")
        return None

    html_code = f"""
    <div id="firebaseui-auth-container"></div>
    <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-app-compat.js"></script>
    <script src="https://www.gstatic.com/firebasejs/9.23.0/firebase-auth-compat.js"></script>
    <script>
      const firebaseConfig = {{
        apiKey: "{api_key}",
        authDomain: "{auth_domain}",
        projectId: "{project_id}",
        appId: "{app_id}"
      }};
      if (!firebase.apps.length) {{
        firebase.initializeApp(firebaseConfig);
      }}

      function signIn() {{
        const provider = new firebase.auth.GoogleAuthProvider();
        firebase.auth().signInWithPopup(provider).then((result) => {{
          const user = result.user;
          const payload = {{
            email: user.email || null,
            displayName: user.displayName || null,
            uid: user.uid || null
          }};
          // Navigate with query param so Streamlit can pick it up
          const qp = new URLSearchParams(window.location.search);
          qp.set('auth', encodeURIComponent(JSON.stringify(payload)));
          window.location.search = qp.toString();
        }}).catch((error) => {{
          console.error(error);
        }});
      }}
    </script>
    <button onclick="signIn()" style="padding:8px 12px;border-radius:6px;border:1px solid #ccc;background:#fff;cursor:pointer">
      Sign in with Google
    </button>
    """
    components.html(html_code, height=80)
    # Read query param if present
    try:
        params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
        auth_param = params.get("auth")
        if not auth_param:
            return None
        # Streamlit may return list; handle both
        if isinstance(auth_param, list):
            auth_param = auth_param[0]
        data = json.loads(json.loads(f'"{auth_param}"'))  # decodeURIComponent effect
        return data
    except Exception:
        return None


def require_auth() -> bool:
    allowed_email = st.secrets.get("auth", {}).get("allowed_email")
    # If no allowed email configured, skip auth for E2E checks
    if not allowed_email:
        return True
    if "auth_user" not in st.session_state:
        st.session_state["auth_user"] = None

    if st.session_state["auth_user"] and allowed_email:
        return st.session_state["auth_user"].get("email") == allowed_email

    st.header("🔐 Sign in")
    st.caption("Sign in with your Google account to continue.")
    data = google_sign_in_widget()
    if data and not data.get("error"):
        st.session_state["auth_user"] = data
        if allowed_email and data.get("email") != allowed_email:
            st.error("This account is not authorized.")
            return False
        st.success("Signed in!")
        # Clear auth param from URL
        try:
            params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
            if hasattr(st, "query_params"):
                if "auth" in st.query_params:
                    del st.query_params["auth"]
            else:
                st.experimental_set_query_params()
        except Exception:
            pass
        return True

    return False

