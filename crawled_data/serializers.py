from rest_framework import serializers
from .models import RecipeData


class RecipeDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecipeData
        fields = ('title', 'link')