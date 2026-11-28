"""Sending the reset and verification mails. Two backends, no new package.

- console (default): the mail is written to the server log, nothing is sent. Good for development.
- smtp: used when RECIPE_SMTP_HOST is set. Settings (environment variables, never in git):
    RECIPE_SMTP_HOST, RECIPE_SMTP_PORT (587), RECIPE_SMTP_USER, RECIPE_SMTP_PASSWORD (secret),
    RECIPE_SMTP_FROM (defaults to the user), RECIPE_SMTP_SSL=1 for port 465.
A mailer never raises into a request: a failed send is logged and the API answers the same way.
"""
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger(__name__)


class ConsoleMailer:
    def send(self, to: str, subject: str, body: str) -> bool:
        log.warning("MAIL (console, not sent) to=%s subject=%s\n%s", to, subject, body)
        return True


class SMTPMailer:
    def __init__(self, host, port=587, user="", password="", sender="", use_ssl=False):
        self.host, self.port, self.user, self.password = host, int(port), user, password
        self.sender, self.use_ssl = sender or user, use_ssl

    def send(self, to: str, subject: str, body: str) -> bool:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        try:
            if self.use_ssl:
                server = smtplib.SMTP_SSL(self.host, self.port, timeout=15, context=ssl.create_default_context())
            else:
                server = smtplib.SMTP(self.host, self.port, timeout=15)
            with server:
                if not self.use_ssl:
                    server.starttls(context=ssl.create_default_context())
                if self.user:
                    server.login(self.user, self.password)
                server.send_message(msg)
            return True
        except Exception:
            log.exception("sending mail to %s failed", to)
            return False


def get_mailer():
    host = os.getenv("RECIPE_SMTP_HOST", "").strip()
    if not host:
        return ConsoleMailer()
    return SMTPMailer(host, os.getenv("RECIPE_SMTP_PORT", "587"), os.getenv("RECIPE_SMTP_USER", ""),
                      os.getenv("RECIPE_SMTP_PASSWORD", ""), os.getenv("RECIPE_SMTP_FROM", ""),
                      os.getenv("RECIPE_SMTP_SSL", "").strip().lower() in ("1", "true", "yes"))


def app_url() -> str:
    return os.getenv("RECIPE_APP_URL", "http://localhost:3000").rstrip("/")


def reset_mail(token: str):
    link = f"{app_url()}/reset-password?token={token}"
    return ("Matbakhak AI: reset your password | إعادة تعيين كلمة المرور",
            f"Open this link to choose a new password (valid 30 minutes):\n{link}\n\n"
            f"افتحي الرابط ده عشان تختاري كلمة مرور جديدة (صالح 30 دقيقة).\n\n"
            f"If you did not ask for this, ignore this mail. | لو مش انتي اللي طلبتي، تجاهلي الرسالة.", link)


def verify_mail(token: str):
    link = f"{app_url()}/verify-email?token={token}"
    return ("Matbakhak AI: confirm your email | تأكيد الإيميل",
            f"Open this link to confirm your email (valid 24 hours):\n{link}\n\n"
            f"افتحي الرابط ده عشان تأكدي الإيميل (صالح 24 ساعة).", link)
