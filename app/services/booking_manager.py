from typing import Dict, Any, Optional, List
from datetime import datetime, time
import json
import uuid
from sqlalchemy.orm import Session
from app.services.email_service import EmailService
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookingManager:
    """Service to manage interview bookings."""
    
    def __init__(self, db: Session, email_service: EmailService):
        """
        Initialize the booking manager.
        
        Args:
            db: Database session
            email_service: Email service for sending confirmations
        """
        self.db = db
        self.email_service = email_service
    
    def process_booking(self, booking_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a booking request and save it to the database.
        
        Args:
            booking_info: Dictionary containing booking information
                (name, email, date, time)
            
        Returns:
            Dictionary with booking status and details
        """
        # Validate booking information
        if not self._validate_booking(booking_info):
            return {
                "success": False,
                "message": "Invalid booking information. Please check your details."
            }
            
        try:
            # Save booking to database
            booking_id = self._save_booking(booking_info)
            
            # Send confirmation email
            email_sent = self.email_service.send_booking_confirmation(
                booking_info["email"], booking_info
            )
            
            return {
                "success": True,
                "booking_id": booking_id,
                "email_sent": email_sent,
                "booking_info": booking_info
            }
        except Exception as e:
            logger.error(f"Error processing booking: {e}")
            return {
                "success": False,
                "message": "An error occurred while processing your booking."
            }
    
    def _validate_booking(self, booking_info: Dict[str, Any]) -> bool:
        """
        Validate booking information.
        
        Args:
            booking_info: Dictionary containing booking information
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["name", "email", "date", "time"]
        
        # Check if all required fields are present and non-empty
        if not all(booking_info.get(field) for field in required_fields):
            return False
            
        # Validate email format (basic check)
        email = booking_info.get("email", "")
        if not "@" in email or not "." in email:
            return False
            
        # Additional validations could be added here:
        # - Date format validation
        # - Time format validation
        # - Check if the requested time slot is available
        
        return True
    
    def _save_booking(self, booking_info: Dict[str, Any]) -> str:
        """
        Save booking information to the database.
        
        Args:
            booking_info: Dictionary containing booking information
            
        Returns:
            Booking ID
        """
        # Generate a booking ID
        booking_id = str(uuid.uuid4())
        
        # Create a new booking record
        booking = {
            "booking_id": booking_id,
            "name": booking_info.get("name"),
            "email": booking_info.get("email"),
            "date": booking_info.get("date"),
            "time": booking_info.get("time"),
            "created_at": datetime.now().isoformat()
        }
        
        # In a real implementation, you would save to your database model
        # For example:
        # db_booking = BookingModel(**booking)
        # self.db.add(db_booking)
        # self.db.commit()
        
        # For now, we'll just log the booking
        logger.info(f"Booking saved: {json.dumps(booking)}")
        
        return booking_id
        
    def get_bookings(self, email: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all bookings or bookings for a specific email.
        
        Args:
            email: Optional email to filter by
            
        Returns:
            List of booking dictionaries
        """
        # In a real implementation, you would query your database
        # For example:
        # query = self.db.query(BookingModel)
        # if email:
        #     query = query.filter(BookingModel.email == email)
        # return [booking.to_dict() for booking in query.all()]
        
        # For now, we'll return an empty list
        return []
        
    def cancel_booking(self, booking_id: str) -> bool:
        """
        Cancel a booking.
        
        Args:
            booking_id: The ID of the booking to cancel
            
        Returns:
            True if canceled successfully, False otherwise
        """
        # In a real implementation, you would delete or mark as canceled in your database
        # For example:
        # booking = self.db.query(BookingModel).filter(BookingModel.booking_id == booking_id).first()
        # if booking:
        #     booking.status = "canceled"
        #     self.db.commit()
        #     return True
        # return False
        
        # For now, we'll just log the cancellation
        logger.info(f"Booking canceled: {booking_id}")
        return True
