from django.db import models

# Create your models here.
from django.db import models


class Recipe(models.Model):
    ingredient = models.CharField(max_length=30)

    def __str__(self):
        return self.ingredient
