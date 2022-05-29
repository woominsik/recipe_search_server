from django.db import models


class Image(models.Model):
    image = models.ImageField(default='media/default_image.jpeg')