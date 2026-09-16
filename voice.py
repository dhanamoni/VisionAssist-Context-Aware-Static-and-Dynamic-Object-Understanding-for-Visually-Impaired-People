import pyttsx3

# Create voice engine
engine = pyttsx3.init()

# Voice settings
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)


def speak(message):
    print("Voice:", message)
    engine.say(message)
    engine.runAndWait()


if __name__ == "__main__":
    speak("VisionAssist voice assistance is working.")