import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from app.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    """Service to handle email sending for booking confirmations."""
    
    def __init__(
        self, 
        smtp_server: str = settings.SMTP_SERVER,
        smtp_port: int = settings.SMTP_PORT,
        smtp_username: str = settings.SMTP_USERNAME,
        smtp_password: str = settings.SMTP_PASSWORD,
        sender_email: str = settings.SENDER_EMAIL
    ):
        """
        Initialize the email service.
        
        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            smtp_username: SMTP authentication username
            smtp_password: SMTP authentication password
            sender_email: Email address to send from
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.smtp_username = smtp_username
        self.smtp_password = smtp_password
        self.sender_email = sender_email
        
    def _format_booking_confirmation(self, booking_info: Dict[str, Any]) -> str:
        """
        Format an email for booking confirmation.
        
        Args:
            booking_info: Dictionary containing booking details
            
        Returns:
            Formatted email body as HTML
        """
        # Format date for better readability if possible
        date_str = booking_info.get("date", "")
        time_str = booking_info.get("time", "")
        name = booking_info.get("name", "")
        
        try:
            # Try to parse and reformat the date for consistency
            # This is a simplified example - in production you'd want more robust date parsing
            parsed_date = datetime.strptime(date_str, "%Y/%m/%d")
            date_formatted = parsed_date.strftime("%A, %B %d, %Y")
        except ValueError:
            # If we can't parse, just use the original string
            date_formatted = date_str
        
        return f"""
        <html>
        <body>
            <h2>Interview Confirmation</h2>
            <p>Dear {name},</p>
            <p>Your interview has been scheduled for:</p>
            <p><strong>Date:</strong> {date_formatted}<br>
            <strong>Time:</strong> {time_str}</p>
            <p>Please arrive 10 minutes early. If you need to reschedule, 
            please contact us at least 24 hours in advance.</p>
            <p>Thank you,<br>
            The Interview Team</p>
        </body>
        </html>
        """
    
    def send_booking_confirmation(
        self, 
        recipient_email: str, 
        booking_info: Dict[str, Any]
    ) -> bool:
        """
        Send a booking confirmation email.
        
        Args:
            recipient_email: Email address of the recipient
            booking_info: Dictionary containing booking details
            
        Returns:
            True if email was sent successfully, False otherwise
        """
        if not recipient_email or not booking_info:
            logger.error("Missing recipient email or booking information")
            return False
        
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = "Your Interview Confirmation"
        message["From"] = self.sender_email
        message["To"] = recipient_email
        
        # Create HTML email body
        html_content = self._format_booking_confirmation(booking_info)
        email_body = MIMEText(html_content, "html")
        message.attach(email_body)
        
        try:
            # Connect to SMTP server and send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.ehlo()
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(self.sender_email, recipient_email, message.as_string())
            
            logger.info(f"Booking confirmation email sent to {recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send booking confirmation email: {e}")
            return False

# Create a mock version for testing without actual email sending
class MockEmailService(EmailService):
    """Mock email service for testing without sending actual emails."""
    
    def __init__(self):
        """Initialize with dummy values."""
        super().__init__(
            smtp_server="localhost",
            smtp_port=25,
            smtp_username="test",
            smtp_password="test",
            sender_email="test@example.com"
        )
        self.sent_emails = []
    
    def send_booking_confirmation(
        self, 
        recipient_email: str, 
        booking_info: Dict[str, Any]
    ) -> bool:
        """
        Mock sending a booking confirmation email.
        
        Args:
            recipient_email: Email address of the recipient
            booking_info: Dictionary containing booking details
            
        Returns:
            Always True, and stores the email in sent_emails
        """
        if not recipient_email or not booking_info:
            logger.error("Missing recipient email or booking information")
            return False
            
        # Create HTML email body for logging
        html_content = self._format_booking_confirmation(booking_info)
        
        # Store the email details for inspection
        self.sent_emails.append({
            "to": recipient_email,
            "subject": "Your Interview Confirmation",
            "body": html_content,
            "booking_info": booking_info
        })
        
        logger.info(f"[MOCK] Booking confirmation email sent to {recipient_email}")
        return True
