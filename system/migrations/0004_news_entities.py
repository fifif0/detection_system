# Generated by Django 4.1.7 on 2023-11-19 18:51

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('system', '0003_alter_news_result'),
    ]

    operations = [
        migrations.AddField(
            model_name='news',
            name='entities',
            field=models.JSONField(default=dict),
        ),
    ]
