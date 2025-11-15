"""
Gesture Recognition Module for Nokia Snake Game
Handles hand tracking and gesture detection using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class GestureController:
    def __init__(self):
        """Initialize MediaPipe hands and gesture detection"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face = mp.solutions.face_detection
        
        # Initialize hands detection
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize face detection
        self.face_detection = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        
        # Gesture state tracking
        self.previous_position = None
        self.gesture_threshold = 0.05
        self.current_direction = None
        self.gesture_cooldown = 0
        self.max_cooldown = 10
        
    def detect_gestures(self, frame: np.ndarray) -> Tuple[Optional[str], bool, np.ndarray]:
        """
        Detect hand gestures and return direction and pinch state
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (direction, is_pinching, annotated_frame)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process hands
        hand_results = self.hands.process(rgb_frame)
        
        # Process face
        face_results = self.face_detection.process(rgb_frame)
        
        # Create annotated frame
        annotated_frame = frame.copy()
        
        direction = None
        is_pinching = False
        
        # Draw face detections
        if face_results.detections:
            for detection in face_results.detections:
                mp.solutions.drawing_utils.draw_detection(annotated_frame, detection)
        
        # Process hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Get hand center position (wrist)
                wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
                current_pos = np.array([wrist.x, wrist.y])
                
                # Detect swipe gestures
                if self.previous_position is not None and self.gesture_cooldown <= 0:
                    movement = current_pos - self.previous_position
                    
                    # Check for significant movement
                    if np.linalg.norm(movement) > self.gesture_threshold:
                        if abs(movement[0]) > abs(movement[1]):
                            # Horizontal movement
                            if movement[0] > 0:
                                direction = "RIGHT"
                            else:
                                direction = "LEFT"
                        else:
                            # Vertical movement
                            if movement[1] > 0:
                                direction = "DOWN"
                            else:
                                direction = "UP"
                        
                        if direction != self.current_direction:
                            self.current_direction = direction
                            self.gesture_cooldown = self.max_cooldown
                
                # Detect pinch gesture (thumb and index finger close)
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                thumb_pos = np.array([thumb_tip.x, thumb_tip.y])
                index_pos = np.array([index_tip.x, index_tip.y])
                
                distance = np.linalg.norm(thumb_pos - index_pos)
                is_pinching = distance < 0.05
                
                # Draw pinch indicator
                if is_pinching:
                    cv2.putText(annotated_frame, "SPEED BOOST!", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
                self.previous_position = current_pos
        
        # Update cooldown
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
        
        # Display current direction
        if self.current_direction:
            cv2.putText(annotated_frame, f"Direction: {self.current_direction}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        return self.current_direction, is_pinching, annotated_frame
    
    def reset_gesture_state(self):
        """Reset gesture detection state"""
        self.previous_position = None
        self.current_direction = None
        self.gesture_cooldown = 0
