import base64
import json
import os
import smtplib
from email.message import EmailMessage

# deployed in Google Cloud Functions
def handle_asl_event(event, context):
    payload = json.loads(
        base64.b64decode(event["data"]).decode("utf-8")
    )

    subject = "ASL Batch Completed"
    recipient = payload.get("recipient_email") 

    msg = EmailMessage()
    msg["From"] = os.environ["GMAIL_USER"]
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(json.dumps(payload, indent=2))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(
            os.environ["GMAIL_USER"],
            os.environ["GMAIL_APP_PASSWORD"]
        )
        server.send_message(msg)
