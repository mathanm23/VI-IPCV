import time

class DrowsinessMonitor:
    def __init__(self, threshold_seconds=3.0):
        """
        Initializes the Drowsiness logic monitor.
        
        Args:
            threshold_seconds: Number of seconds eyes must be closed to trigger DROWSY state.
        """
        self.threshold_seconds = threshold_seconds
        self.closed_start_time = None
        self.elapsed_closed_time = 0.0
        self.status = "AWAKE"

    def update_state(self, left_eye_state, right_eye_state):
        """
        Updates the drowsiness status based on the current frame's eye states.
        
        Args:
            left_eye_state: "Open" or "Closed"
            right_eye_state: "Open" or "Closed"
            
        Returns:
            status: "AWAKE" or "DROWSY"
        """
        # Logic: Both eyes must be closed to consider the person drowsy.
        # This prevents a wink or a single-eye misclassification from triggering the timer.
        is_currently_closed = (left_eye_state == "Closed" and right_eye_state == "Closed")
        
        if is_currently_closed:
            if self.closed_start_time is None:
                self.closed_start_time = time.time()
            self.elapsed_closed_time = time.time() - self.closed_start_time
        elif left_eye_state == "Open" or right_eye_state == "Open":
            # If at least one eye is open, they are awake. Reset the timer.
            self.closed_start_time = None
            self.elapsed_closed_time = 0.0
        else:
            # Handle "Unknown" states
            pass

        # Check against time threshold
        if self.elapsed_closed_time >= self.threshold_seconds:
            self.status = "DROWSY"
        else:
            self.status = "AWAKE"
            
        return self.status
