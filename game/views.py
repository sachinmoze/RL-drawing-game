from django.shortcuts import render
from uuid import uuid4

def index(request):
    return render(request, 'game/index.html')

def room(request, room_name):
    username = request.GET.get('username', 'Anonymous')
    user_id = request.GET.get('user_id')
    if not user_id:
        user_id = str(uuid4())
    return render(request, 'game/room.html', {
        'room_name': room_name,
        'username': username,
        'user_id': user_id
    })
