import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import time
from datetime import datetime

# --- IMPORT YOUR PROJECT FILES ---
# NOTE: Ensure 'gesture_controller.py' and 'snake_game.py' are in the same folder.
# NOTE: You MUST modify SnakeGame.draw() to return an OpenCV image (NumPy array).
# from snake_game import SnakeGame 
# from gesture_controller import GestureController 
# --- PLACEHOLDERS for development ---
class MockSnakeGame:
    def __init__(self):
        self.game_over = False
        self.score = 0
        self.current_direction = "RIGHT"
    def change_direction(self, direction): self.current_direction = direction
    def set_speed_boost(self, boost): pass
    def update(self): 
        if self.game_over: return
        self.score += 1
        if self.score > 20: self.game_over = True
    def handle_restart(self, gesture):
        if self.game_over and gesture == "UP": self.game_over = False; self.score = 0; self.current_direction = "RIGHT"
    def draw(self):
        # This function MUST be rewritten to return a BGR NumPy array (image)
        # instead of drawing with Pygame.
        H, W = 480, 640
        game_img = np.zeros((H, W, 3), dtype=np.uint8) 
        cv2.putText(game_img, f"Score: {self.score}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(game_img, f"Dir: {self.current_direction}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if self.game_over:
            cv2.putText(game_img, "GAME OVER", (W//2 - 100, H//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        return game_img
    
class MockGestureController:
    def detect_gestures(self, frame):
        # Mock detection: always says 'RIGHT'
        H, W, _ = frame.shape
        cv2.putText(frame, "Mock Gesture: RIGHT", (10, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        return "RIGHT", False, frame
# --- END PLACEHOLDERS ---

# 1. VIDEO PROCESSOR CLASS (Replaces GestureController Thread)
# This runs in a separate thread handled by streamlit-webrtc.
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize your SnakeGame and GestureController outside Streamlit's main script flow
        # Use st.session_state to pass the Game object from the main script
        # NOTE: Using a simple time-based lock here, but for production, use Streamlit's session_state
        self.game_state = st.session_state.snake_game 
        # self.gesture_detector = GestureController() # Use your actual class
        self.gesture_detector = MockGestureController() # Using mock for demo
        self.last_game_update = time.time()
        self.fps = 10 # Base game tick rate

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert PyAV frame to OpenCV NumPy array
        img = frame.to_ndarray(format="bgr24")

        # 1. DETECT GESTURE (Your original logic)
        gesture, is_speed_boost, annotated_frame = self.gesture_detector.detect_gestures(img)

        # 2. UPDATE GAME STATE
        # This logic is critical for the game to update continuously independent of
        # Streamlit's main script reruns.
        
        # Lock required to safely access session_state or shared variables
        # Since we are using session_state here, we rely on Streamlit's implicit thread safety
        # and accept the limitations of st.session_state inside webrtc thread.
        # A safer pattern is to use a thread-safe Queue/Lock, but we'll stick to 
        # the simple update for demonstration.

        self.game_state.change_direction(gesture)
        self.game_state.set_speed_boost(is_speed_boost)
        
        # Calculate update rate (target_fps should come from self.game_state if it changes)
        target_fps = self.game_state.fps if hasattr(self.game_state, 'fps') else self.fps 
        
        current_time = time.time()
        if current_time - self.last_game_update >= 1.0 / target_fps:
             self.game_state.update()
             self.last_game_update = current_time

        # 3. Handle Game Over Restart
        if self.game_state.game_over:
             self.game_state.handle_restart(gesture)
        
        # Return the annotated frame to be displayed in the Streamlit component
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# 2. STREAMLIT MAIN APP FUNCTION (Replaces run() and main())
def main():
    st.set_page_config(page_title="Gesture Snake", layout="wide")
    st.title("🐍 Nokia Snake Game with Gesture Control")

    # Initialize the game state in Streamlit's session state
    # This prevents the game from resetting on every rerun
    if 'snake_game' not in st.session_state:
        # st.session_state.snake_game = SnakeGame() # Use your actual class
        st.session_state.snake_game = MockSnakeGame() # Using mock for demo
        st.session_state.snake_placeholder = st.empty() # Placeholder for the game board image

    col_video, col_game = st.columns([1, 1])

    with col_video:
        st.header("Gesture Recognition (Webcam)")
        # 3. Start the WebRTC Streamer
        # The GestureProcessor will run its logic (detection & game update) here
        webrtc_streamer(
            key="gesture-control",
            video_processor_factory=GestureProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True # Use async processing for better performance
        )
        st.info("Look at your camera to control the snake. Move your hand up/down/left/right.")

    with col_game:
        st.header("Game Board")
        # Display the game score and state
        score_display = st.empty()
        
        # 4. Display the Game Board Image
        # Since the game state is updated in the webrtc thread, the main script 
        # needs a way to constantly refresh the display. A simple loop is used here.
        # NOTE: The game is updated in the VideoProcessor, not this loop!

        # Use st.empty() from session state to draw the game image
        snake_placeholder = st.session_state.snake_placeholder 
        
        # This loop keeps the Streamlit app responsive and updates the display
        while True:
            # Check for necessary imports before proceeding
            # if 'SnakeGame' not in globals(): # Use this check when integrating your classes
            #    score_display.markdown("### ⚠️ Waiting for game state (Check logs for errors)")
            #    time.sleep(1)
            #    continue

            # 1. Update Score/Status Display
            current_game = st.session_state.snake_game
            score_display.markdown(f"### Score: **{current_game.score}** | Status: **{'Game Over!' if current_game.game_over else 'Playing'}**")
            
            # 2. Get the game image from the modified SnakeGame.draw()
            game_frame = current_game.draw()
            
            # 3. Draw the game image to the placeholder
            snake_placeholder.image(game_frame, channels="BGR", use_column_width=True)
            
            # Wait a short moment to avoid overwhelming the CPU
            time.sleep(1/30) # Refresh the image at ~30 FPS

# Call the main function
if __name__ == "__main__":
    # Ensure you add 'av' and 'streamlit-webrtc' to your requirements.txt!
    # Also ensure you use the config.toml file for Python 3.10 as discussed.
    main()
