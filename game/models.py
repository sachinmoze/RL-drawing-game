from django.db import models

class Room(models.Model):
    name = models.CharField(max_length=255, unique=True)
    users = models.PositiveIntegerField(default=0)
    current_drawer = models.PositiveIntegerField(default=0)
    current_word = models.CharField(max_length=255, blank=True, null=True)  # Add current_word to store the word to be drawn

    def __str__(self):
        return self.name

class User(models.Model):
    username = models.CharField(max_length=255)
    user_id = models.CharField(max_length=255)
    room = models.ForeignKey(Room, related_name='user_set', on_delete=models.CASCADE)

    def __str__(self):
        return self.username
