
import pyttsx3

def speak(message):
    print("Voice:", message)

    # Create a fresh voice engine for each message
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)

    engine.say(str(message))
    engine.runAndWait()
    engine.stop()


if __name__ == "__main__":
    speak("VisionAssist voice assistance is working.")
