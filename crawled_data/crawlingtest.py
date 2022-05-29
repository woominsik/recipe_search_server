from unicodedata import normalize
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Banana.settings")
import django
django.setup()


from crawled_data.models import RecipeData

# clien => client
def crawling_data(ingredient):
    output = []
    for i in range(len(ingredient)):
        ingredient[i]= ingredient[i].lower()
    igdDict = {'potato': 'basefoods%5B%5D=851', 'meat': 'basefoods%5B%5D=2489', 'onion': 'basefoods%5B%5D=1639',
               'egg': 'basefoods%5B%5D=2781', \
               'carrot': 'basefoods%5B%5D=1392', 'cabbage': 'basefoods%5B%5D=1625',
              }

    recipe = list()
    recipeAddress = list()
    result = list()
    resultAddress = list()

    def getRecipeAddress(ingredient):
        if isinstance(ingredient, str):
            address = 'https://haemukja.com/recipes?'
            pageNum = 1
            address += igdDict.get(ingredient)
            address += '&page=' + str(pageNum) + '&sort=rnk'

        # 크롤링할 사이트의 주소 초기값 설정
        else:
            address = 'https://haemukja.com/recipes?'

            # 재료에 따른 주소값 변경
            address += igdDict.get(ingredient[0])
            for s in ingredient[1:]:
                address += '&' + igdDict.get(s)

        return address

    def crawl(address, recipeNum):
        pageNum = 1
        tempaddress = address + '&page='
        address += '&page=' + str(pageNum) + '&sort=rnk'
        # 해당 재료들로 만들 수 있는 요리들 크롤링
        recipe.clear()
        recipeAddress.clear()
        address += (str(pageNum) + '&sort=rnk')
        response = urlopen(address)
        soup = BeautifulSoup(response, 'html.parser')
        for j in range(1, 13):
            selector = '#content > section > div.recipes > div > ul > li:nth-child(' + str(j) + ') > p > a > strong'
            selector2 = '#content > section > div.recipes > div > ul > li:nth-child(' + str(
                j) + ') > div.share_on > dl > dd:nth-child(5) > a'
            title = soup.select_one(selector)
            address = soup.select_one(selector2)
            if not title:
                break
            recipeAddress.append(str(address))
            recipe.append(title.string)
        address = tempaddress
        pageNum = pageNum + 1

        for i in range(len(recipeAddress)):
            startAddress = recipeAddress[i].find("http")
            recipeAddress[i] = recipeAddress[i][startAddress:]
            endAddress = recipeAddress[i].find("\"")
            recipeAddress[i] = recipeAddress[i][:endAddress]

        # 영어가 들어가는 레시피의 경우 중복이 일어나는 경우가 발생하여 영어가 들어간 레시피는 삭제
        for i in recipe:
            if 'a' <= i[0].lower() <= "z":
                a = recipe.index(i)
                recipe.remove(i)
                recipeAddress.remove(recipeAddress[a])

        for i in range(recipeNum):
            if recipe:
                result.append(recipe[i])
                resultAddress.append(recipeAddress[i])

        # return recipe

    # 재료 입력
    # print('재료를 입력하세요 : ')
    # ingredient=list(input().split())

    # 각 재료에 해당하는 주소
    address = getRecipeAddress(ingredient)

    # 주소에 해당하는 레시피 출력
    crawl(address, 4)

    # 레시피가 없을 때 or연산으로
    if len(recipe) == 0:
        outputLen = 4
        restNum = len(ingredient)
        for i in range(len(ingredient)):
            address = getRecipeAddress(ingredient[i])
            num = int(outputLen / restNum)
            crawl(address, num)
            outputLen = outputLen - num
            restNum = restNum - 1
            
    for i in range(len(resultAddress)):
        a=resultAddress[i].find("/recipes")
        resultAddress[i]=resultAddress[i][a:]
        resultAddress[i]="https://m.haemukja.com/#"+resultAddress[i]
    
    output = {}
    for i in range(len(result)):
        output[result[i]] = resultAddress[i]
    return output
    

queryset = RecipeData.objects.all()
queryset.delete()


ingredient = ["감자"]
# print(crawling_data(ingredient))

if __name__=='__main__':
    blog_data_dict = crawling_data(ingredient)
    #print(blog_data_dict.items())
    for t, l in blog_data_dict.items():
        #print(t, l)
        RecipeData(title=t, link=l).save()











