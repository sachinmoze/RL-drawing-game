import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import asyncio
import uuid
from .models import Room, User
from .q_learning_agent import QLearningAgent
import gym
from .rl_model import choose_word, suggest_steps, provide_suggestions, check_guess, update_model, get_state, adjust_word_difficulty

class GameConsumer(AsyncWebsocketConsumer):
    env = gym.make('CartPole-v1')
    agent = QLearningAgent(env)  # Initialize the Q-learning agent with the environment

    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'game_{self.room_name}'
        self.user_id = str(uuid.uuid4())
        self.user_name = None
        self.current_word = None  # Add a variable to store the current word
        self.env.reset()  # Reset the environment upon connection

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        # Accept the connection
        await self.accept()

    async def disconnect(self, close_code):
        # Remove user from room
        if self.user_id:
            await self.remove_user_from_room()

        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')

        if action == 'join':
            self.user_name = data['username']
            self.user_id = data['userId']
            await self.add_user_to_room(self.user_name, self.user_id)
        elif action == 'start_game':
            self.env.reset()  # Reset the environment when the game starts
            await self.start_game()
        elif action == 'drawing':
            drawing = data['drawing']
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'draw',
                    'drawing': drawing,
                    'drawer': data['drawer'],
                    'suggestion': provide_suggestions(drawing)
                }
            )
            state = get_state(drawing)
            action = self.agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            next_state = get_state(next_state)
            self.agent.update_q_table(state, action, reward, next_state)
        elif action == 'guess':
            correct = await sync_to_async(check_guess)(self.room_name, data['guess'])
            if correct:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'correct_guess',
                        'username': data['username'],
                        'guess': data['guess']
                    }
                )
                # Adjust difficulty if the guess is correct
                if self.current_word:
                    adjust_word_difficulty(self.current_word, correct_guess=True)
            else:
                if self.current_word:
                    adjust_word_difficulty(self.current_word, correct_guess=False)
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'chat_message',
                    'username': data['username'],
                    'message': data['guess']
                }
            )
            await sync_to_async(update_model)(self.room_name, data)

    async def new_word(self, event):
        self.current_word = event['word']  # Store the current word
        await self.send(text_data=json.dumps({
            'action': 'new_word',
            'word': event['word'],
            'steps': event['steps'],
            'drawer': event['drawer']  # Include drawer information
        }))

    async def draw(self, event):
        await self.send(text_data=json.dumps({
            'action': 'draw',
            'drawing': event['drawing'],
            'drawer': event['drawer'],
            'suggestion': event['suggestion'],
        }))

    async def chat_message(self, event):
        await self.send(text_data=json.dumps({
            'action': 'chat_message',
            'username': event['username'],
            'message': event['message'],
        }))

    async def correct_guess(self, event):
        await self.send(text_data=json.dumps({
            'action': 'correct_guess',
            'username': event['username'],
            'guess': event['guess']
        }))

    async def start_turn(self, drawer):
        word = choose_word(self.room_name)
        steps = suggest_steps(word)

        # Update the current word for the room
        await sync_to_async(Room.objects.filter(name=self.room_name).update)(current_word=word)
        self.current_word = word  # Store the current word
        print(f"Drawer: {drawer}, Word: {word}")
        # Notify the drawer
        await self.channel_layer.send(
            drawer,
            {
                'type': 'new_word',
                'word': word,
                'steps': steps,
                'drawer': drawer  # Include drawer information
            }
        )

        # Notify others
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'turn',
                'drawer': drawer,
                'word': word,
                'steps': steps
            }
        )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'clear_canvas'
            }
        )
        await asyncio.sleep(60)
        await self.next_turn()

    async def turn(self, event):
        await self.send(text_data=json.dumps({
            'action': 'turn',
            'drawer': event['drawer'],
            'word': event['word'],
            'steps': event['steps']
        }))

    async def clear_canvas(self, event):
        await self.send(text_data=json.dumps({
            'action': 'clear_canvas'
        }))

    async def next_turn(self):
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = (room.current_drawer + 1) % room.users
        await sync_to_async(room.save)()
        drawer = room.current_drawer
        drawer_user = await self.get_user_by_turn(drawer)
        await self.start_turn(drawer_user)

    @sync_to_async
    def add_user_to_room(self, username, user_id):
        room, created = Room.objects.get_or_create(name=self.room_name)
        User.objects.create(username=username, user_id=user_id, room=room)
        room.users += 1
        room.save()

    @sync_to_async
    def remove_user_from_room(self):
        room = Room.objects.get(name=self.room_name)
        users = User.objects.filter(user_id=self.user_id, room=room)
        for user in users:
            user.delete()
        room.users = max(0, room.users - len(users))
        room.save()

    @sync_to_async
    def update_user_count(self, increment=True):
        room, created = Room.objects.get_or_create(name=self.room_name)
        if increment:
            room.users += 1
        else:
            room.users -= 1
        room.save()

    @sync_to_async
    def get_user_count(self):
        room, created = Room.objects.get_or_create(name=self.room_name)
        return room.users

    @sync_to_async
    def get_user_by_turn(self, turn):
        room = Room.objects.get(name=self.room_name)
        users = list(room.user_set.all())
        if users:
            return users[turn % len(users)].user_id
        return None

    async def broadcast_user_count(self):
        user_count = await self.get_user_count()
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'user_count',
                'user_count': user_count
            }
        )

    async def user_count(self, event):
        await self.send(text_data=json.dumps({
            'action': 'user_count',
            'user_count': event['user_count'],
        }))

    async def send_user_count(self, user_count):
        await self.send(text_data=json.dumps({
            'action': 'user_count',
            'user_count': user_count,
        }))

    async def start_game(self):
        self.env.reset()  # Reset the environment when the game starts
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = 0
        await sync_to_async(room.save)()
        drawer_user = await self.get_user_by_turn(room.current_drawer)
        await self.start_turn(drawer_user)

    async def add_user_to_room(self, username, user_id):
        await self._add_user_to_room(username, user_id)
        await self.broadcast_user_count()

    async def remove_user_from_room(self):
        await self._remove_user_from_room()
        await self.broadcast_user_count()

    @sync_to_async
    def _add_user_to_room(self, username, user_id):
        room, created = Room.objects.get_or_create(name=self.room_name)
        User.objects.create(username=username, user_id=user_id, room=room)
        room.users += 1
        room.save()

    @sync_to_async
    def _remove_user_from_room(self):
        room = Room.objects.get(name=self.room_name)
        users = User.objects.filter(user_id=self.user_id, room=room)
        for user in users:
            user.delete()
        room.users = max(0, room.users - len(users))
        room.save()

    def calculate_reward(self, drawing, guess):
        # Define your reward function based on the drawing and guessing accuracy
        # For simplicity, let's assume a fixed reward
        return 1  # You can customize this based on the actual requirements
