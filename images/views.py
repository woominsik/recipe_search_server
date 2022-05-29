from rest_framework import viewsets
from .serializers import ImageSerializer
from .models import Image
from django.http import JsonResponse
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework import status
#from detection2 import get_ingredients
from category_detection import get_categories
import json
from crawled_data.crawlingtest import crawling_data
import os

IMAGE_PATH = '/workspace/Main_Server/recipe_search_server/media/food.png'

class ImageViewSet(viewsets.ModelViewSet):
    queryset = Image.objects.all()
    serializer_class = ImageSerializer
    
    def custom_create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        print(request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        
        
        # 1. 모델 돌리기 => 출력 : 재료리스트
        '''
        ingredients = get_ingredients(IMAGE_PATH)
        if len(ingredients) == 0:
            os.remove(IMAGE_PATH)
            return Response(serializer.data, status=status.HTTP_400_BAD_REQUEST, headers=headers)
        
        if len(ingredients) > 4:
            ingredients = ingredients[0:4]
        print('ingredients : ', ingredients)
        '''
        
        categories = get_categories()
        print('categories : ' + categories)

        
        # 2. 크롤링 => 출력 : 음식, (링크1)
        os.remove(IMAGE_PATH)
        result_dict = crawling_data(ingredients)
        '''
        result_dict = {'치즈포테이토': 'https://m.haemukja.com/#/recipes/5397', '감자어묵조림': 'https://m.haemukja.com/#/recipes/4088', ' 치즈 베이컨 포테이토': 'https://m.haemukja.com/#/recipes/3656', '감자 조림': 'https://m.haemukja.com/#/recipes/4081'}
        '''        
        
        # 3. 음식, 링크1를 serializer.data에 넣어주기
        json_object = {
            'foods': []
        }
        for key in result_dict.keys():
            food = {}
            print('food name : ', key)
            food['name'] = key
            food['url'] = result_dict[key]
            json_object['foods'].append(food)
            
        json_result = json.dumps(json_object)
        
        
        #return JsonResponse(json_object, json_dumps_params = {'ensure_ascii': True})
        return Response(json_object, status=status.HTTP_201_CREATED, headers=headers)
    
    def create(self, request):
        #ret = super().create(request)
        ret = self.custom_create(request)
        
        return ret
        
    

