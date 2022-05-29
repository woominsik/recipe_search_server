from django.db import models


# Create your models here.
class RecipeData(models.Model):
    title = models.CharField(max_length=300)
    link = models.URLField()

    def __str__(self):
        return self.title
