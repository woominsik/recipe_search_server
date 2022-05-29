from rest_framework import serializers
from .models import Image


class ImageSerializer(serializers.HyperlinkedModelSerializer):
    image = serializers.ImageField(use_url=True)

    class Meta:
        model = Image
        fields = "__all__"
