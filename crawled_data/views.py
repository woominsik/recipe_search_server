from django.shortcuts import render
from rest_framework import viewsets
from .serializers import RecipeDataSerializer
from .models import RecipeData
# Create your views here.


class CrawledViewSet(viewsets.ModelViewSet):
    queryset = RecipeData.objects.all()
    serializer_class = RecipeDataSerializer