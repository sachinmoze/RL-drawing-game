import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .models import Room
from .rl_model import choose_word, suggest_steps, provide_suggestions, check_guess, update_model
import asyncio
import uuid

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'game_{self.room_name}'
        self.user_id = str(uuid.uuid4())

        # Join room group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )

        # Track the number of users
        await self.update_user_count(increment=True)

        # Accept the connection
        await self.accept()

        # Check if there are enough users to start the game
        user_count = await self.get_user_count()
        await self.send_user_count(user_count)
        if user_count == 2:
            await self.start_game()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

        # Decrement user count
        await self.update_user_count(increment=False)

        user_count = await self.get_user_count()
        await self.send_user_count(user_count)

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')

        if action == 'start_game':
            await self.start_game()
        elif action == 'drawing':
            suggestions = provide_suggestions(data['drawing'])
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'draw',
                    'drawing': data['drawing'],
                    'suggestions': suggestions,
                    'drawer': data['drawer'],
                }
            )
        elif action == 'guess':
            correct = check_guess(self.room_name, data['guess'])
            if correct:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        'type': 'correct_guess',
                        'username': data['username'],
                    }
                )
            update_model(self.room_name, data)

    async def new_word(self, event):
        await self.send(text_data=json.dumps({
            'action': 'new_word',
            'word': event['word'],
            'steps': event['steps'],
        }))

    async def draw(self, event):
        await self.send(text_data=json.dumps({
            'action': 'draw',
            'drawing': event['drawing'],
            'suggestions': event['suggestions'],
            'drawer': event['drawer'],
        }))

    async def correct_guess(self, event):
        await self.send(text_data=json.dumps({
            'action': 'correct_guess',
            'username': event['username'],
        }))

    async def start_turn(self, drawer):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'turn',
                'drawer': drawer,
            }
        )
        await asyncio.sleep(60)
        await self.next_turn()

    async def turn(self, event):
        await self.send(text_data=json.dumps({
            'action': 'turn',
            'drawer': event['drawer'],
        }))

    async def next_turn(self):
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = (room.current_drawer + 1) % room.users
        await sync_to_async(room.save)()
        drawer = room.current_drawer
        await self.start_turn(drawer)

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

    async def send_user_count(self, user_count):
        await self.send(text_data=json.dumps({
            'action': 'user_count',
            'user_count': user_count,
        }))

    async def start_game(self):
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = 0
        await sync_to_async(room.save)()
        word = choose_word(self.room_name)
        steps = suggest_steps(word)
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'new_word',
                'word': word,
                'steps': steps,
            }
        )
        await self.start_turn(room.current_drawer)
