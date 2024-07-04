import json
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import asyncio
import uuid

from .models import Room, User
from .rl_model import choose_word, suggest_steps, check_guess, update_model

class GameConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.room_name = self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = f'game_{self.room_name}'
        self.user_id = self.scope['session'].get('user_id', str(uuid.uuid4()))
        self.user_name = self.scope['session'].get('username', None)

        self.scope['session']['user_id'] = self.user_id
        self.scope['session']['username'] = self.user_name
        await sync_to_async(self.scope['session'].save)()

        await self.channel_layer.group_add(self.room_group_name, self.channel_name)
        await self.accept()

        if self.user_name:
            await self.add_user_to_room(self.user_name, self.user_id)
            await self.broadcast_user_count()

    async def disconnect(self, close_code):
        await self.remove_user_from_room()
        await self.broadcast_user_count()
        await self.channel_layer.group_discard(self.room_group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        action = data.get('action')

        if action == 'join':
            await self.handle_join(data)
        elif action == 'start_game':
            await self.start_game()
        elif action == 'drawing':
            await self.handle_drawing(data)
        elif action == 'guess':
            await self.handle_guess(data)

    async def handle_join(self, data):
        self.user_name = data['username']
        self.user_id = data['userId']
        await self.add_user_to_room(self.user_name, self.user_id)
        await self.broadcast_user_count()

    async def handle_drawing(self, data):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'draw',
                'drawing': data['drawing'],
                'drawer': data['drawer'],
            }
        )

    async def handle_guess(self, data):
        correct = check_guess(self.room_name, data['guess'])
        if correct:
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    'type': 'correct_guess',
                    'username': data['username'],
                    'guess': data['guess']
                }
            )
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'chat_message',
                'username': data['username'],
                'message': data['guess']
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
            'drawer': event['drawer'],
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

        await self.channel_layer.send(
            drawer,
            {
                'type': 'new_word',
                'word': word,
                'steps': steps,
            }
        )

        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'turn',
                'drawer': drawer,
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
        }))

    async def clear_canvas(self, event):
        await self.send(text_data=json.dumps({
            'action': 'clear_canvas'
        }))

    async def next_turn(self):
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = (room.current_drawer + 1) % await sync_to_async(room.user_set.count)()
        if room.current_drawer == 0:
            await self.end_game()
        else:
            await sync_to_async(room.save)()
            drawer = room.current_drawer
            drawer_user = await self.get_user_by_turn(drawer)
            await self.start_turn(drawer_user)

    async def end_game(self):
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'game_end'
            }
        )

    async def game_end(self, event):
        await self.send(text_data=json.dumps({
            'action': 'game_end'
        }))

    @sync_to_async
    def add_user_to_room(self, username, user_id):
        room, created = Room.objects.get_or_create(name=self.room_name)
        user, user_created = User.objects.get_or_create(user_id=user_id, defaults={'username': username, 'room': room})
        if user_created:
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

    async def start_game(self):
        room = await sync_to_async(Room.objects.get)(name=self.room_name)
        room.current_drawer = 0
        await sync_to_async(room.save)()
        drawer_user = await self.get_user_by_turn(room.current_drawer)
        await self.start_turn(drawer_user)
