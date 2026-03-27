# ================= tts_player.py =================
import pyttsx3

def announce(text):
    """
    Speak the given text and block until finished.
    This ensures the full announcement is played before the script continues.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)  # Speech speed
        engine.say(text)
        engine.runAndWait()  # <-- blocks until speech finishes
        engine.stop()
    except Exception as e:
        print("[TTS ERROR]", e)