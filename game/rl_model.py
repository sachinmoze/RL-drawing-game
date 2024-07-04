import random
import base64
import numpy as np
from PIL import Image
import io
from .models import Room
from .q_learning_agent import QLearningAgent
import gym

agent = QLearningAgent(gym.make('CartPole-v1'))  # Initialize the Q-learning agent

def choose_word(room_name):
    return agent.choose_word()

def adjust_word_difficulty(word, correct_guess):
    reward = 1 if correct_guess else -1
    agent.adjust_word_difficulty(word, reward)

def suggest_steps(word):
    return [f"Step 1: Draw the {word}", f"Step 2: Add details to the {word}"]

def provide_suggestions(drawing):
    return "Try to add more details to your drawing."

def check_guess(room_name, guess):
    room = Room.objects.get(name=room_name)
    return room.current_word.lower() == guess.lower()

def update_model(room_name, data):
    drawing = data.get('drawing')
    if not drawing:
        print("No drawing data available for model update.")
        return  # Skip the update if no drawing data

    state = get_state(drawing)
    action = agent.choose_action(state)
    next_state = agent.simulate_next_state(state, action, drawing)
    reward = agent.calculate_reward(drawing, data.get('guess'))
    agent.update_q_table(state, action, reward, next_state)


def simulate_next_state(state, action, drawing):
    # Convert the drawing to a state
    current_state = list(state)  # Convert to list to modify

    if action == 0:  # Draw a line
        # Simulate drawing a line by modifying the state (placeholder logic)
        current_state[0] += 1
    elif action == 1:  # Add a detail
        # Simulate adding a detail (placeholder logic)
        current_state[1] += 1
    elif action == 2:  # Change color
        # Simulate changing color (placeholder logic)
        current_state[2] += 1

    return tuple(current_state)  # Convert back to tuple

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
