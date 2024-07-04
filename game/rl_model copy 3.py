import random
import base64
import numpy as np
from PIL import Image
import io
from .models import Room

def choose_word(room_name):
    words = ["cat", "dog", "car", "tree", "house"]
    return random.choice(words)

def suggest_steps(word):
    steps = {
        "cat": ["Draw a circle for the head", "Add ears and eyes", "Draw the body", "Add the legs and tail"],
        "dog": ["Draw a circle for the head", "Add ears and eyes", "Draw the body", "Add the legs and tail"],
        "car": ["Draw a rectangle for the body", "Add wheels", "Draw windows and doors"],
        "tree": ["Draw the trunk", "Add branches", "Draw leaves"],
        "house": ["Draw a square for the body", "Add a triangle for the roof", "Draw windows and a door"],
    }
    return steps[word]

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
