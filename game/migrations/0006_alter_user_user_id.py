# Generated by Django 5.0.6 on 2024-07-02 19:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('game', '0005_user_user_id'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='user_id',
            field=models.CharField(max_length=255, unique=True),
        ),
    ]
