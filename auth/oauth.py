import os
from authlib.integrations.starlette_client import OAuth
from starlette.requests import Request

oauth = OAuth()

oauth.register(
    name="google",
    client_id=os.environ["GOOGLE_CLIENT_ID"],
    client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile"},
)

def get_roles(userinfo: dict) -> list[str]:
    email = (userinfo.get("email") or "").strip().lower()

    admin_emails = {
        e.strip().lower()
        for e in os.environ.get("ADMIN_EMAILS", "").split(",")
        if e.strip()
    }

    if email and email in admin_emails:
        return ["admin"]
    return ["user"]

async def oauth_login(request: Request):
    return await oauth.google.authorize_redirect(
        request,
        redirect_uri=request.url_for("auth_callback"),
    )

async def oauth_callback(request: Request):
    token = await oauth.google.authorize_access_token(request)
    return token["userinfo"]

