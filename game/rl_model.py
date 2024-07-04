import random
import base64
import numpy as np
from PIL import Image
import io
from .models import Room

def choose_word(room_name):
    words = ["apple", "banana", "cat", "dog", "elephant"]
    return random.choice(words)

def suggest_steps(word):
    return [f"Step 1: Draw the {word}", f"Step 2: Add details to the {word}"]

def provide_suggestions(drawing):
    return "Try to add more details to your drawing."

def check_guess(room_name, guess):
    room = Room.objects.get(name=room_name)
    return room.current_word.lower() == guess.lower()

def update_model(room_name, data):
    pass

def get_state(drawing):
    if isinstance(drawing, str):
        # Convert the base64 drawing to a grayscale array
        image_data = base64.b64decode(drawing.split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = image.resize((28, 28))  # Resize for consistency
        state = np.array(image).flatten()  # Convert to a 1D array
    elif isinstance(drawing, np.ndarray):
        # If drawing is already a NumPy array
        state = drawing
    else:
        raise ValueError("Unsupported drawing format")
    
    return tuple(state[:4])  # Return only the first 4 elements to match the CartPole state space
