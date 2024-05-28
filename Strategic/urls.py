"""
URL configuration for Strategic project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from Profiling import views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('homepage/recommendation',views.combine_recommendations,name='recommendation'),
    path('homepage/erack/',views.erack,name='erack'),
    path('homepage/crack/',views.crack,name='crack'),
    path('homepage/left/',views.left,name='left'),
    path('homepage/right/',views.right,name='right'),
    path('homepage/brack/',views.brack,name='brack'),
    path('homepage/',views.homepage,name='homepage'),
    path('homepage/menu',views.recommend_menu,name='menu'),
    path('homepage/detector',views.age_gender_detector,name='detector'),
    path('homepage/off',views.off,name='off'),
    path('homepage/season',views.season,name='season'),
    path('homepage/winter',views.winter,name='winter'),
    path('homepage/summer',views.summer,name='summer'),
    path('homepage/spring',views.spring,name='spring'),
    path('homepage/fall',views.fall,name='fall'),
    path('homepage/placement',views.placement,name='placement'),
    path('homepage/add_to_inventory',views.add_to_inventory,name='add_to_inventory'),
    path('homepage/add_to_inventory_season',views.add_to_inventory_season,name='add_to_inventory_season'),
    path('homepage/add_to_inventory_back',views.add_to_inventory_back,name='add_to_inventory_back'),
    path('homepage/inventory',views.inventory,name='inventory'),
    path('homepage/payment',views.payment,name='payment'),
    path('homepage/inside_payment',views.inside_payment,name='inpayment'),
    path('homepage/success',views.success,name='success'),
    path('homepage/aboutus',views.aboutus,name='aboutus'),
    path('homepage/delete',views.delete,name='delete'),
    path('homepage/tanktop',views.tanktop,name='tanktop'),
    path('homepage/skirts',views.skirts,name='skirts'),
    path('homepage/strawhats',views.strawhats,name='strawhats'),
    path('homepage/shorts',views.shorts,name='shorts'),
    path('homepage/strawhats',views.strawhats,name='strawhats'),
    path('homepage/brightcoloredclothing',views.brightcoloredclothing,name='brightcoloredclothing'),
    path('homepage/belts',views.belts,name='belts'),
    path('homepage/canvas',views.canvas,name='canvas'),
    path('homepage/Breathablefabric',views.Breathablefabric,name='Breathablefabric'),
    path('homepage/sandals',views.sandals,name='sandals'),
    path('homepage/flipflops',views.flipflops,name='flipflops'),
    path('homepage/sunglasses',views.sunglasses,name='sunglasses'),
    path('homepage/scarves',views.scarves,name='scarves'),
    path('homepage/coverups',views.coverups,name='coverups'),
    path('homepage/pastel',views.pastel,name='pastel'),
    path('homepage/lightjacket',views.lightjacket,name='lightjacket'),
    path('homepage/floral',views.floral,name='floral'),
    path('homepage/denim',views.denim,name='denim'),
    path('homepage/printedskirts',views.printedskirts,name='printedskirts'),
    path('homepage/tshirts',views.tshirts,name='tshirts'),
    path('homepage/knitwear',views.knitwear,name='knitwear'),
    path('homepage/crossbody',views.crossbody,name='crossbody'),
    path('homepage/whitesneakers',views.whitesneakers,name='whitesneakers'),
    path('homepage/jewel',views.jewel,name='jewel'),
    path('homepage/sweater',views.sweater,name='sweater'),
    path('homepage/boots',views.boots,name='boots'),
    path('homepage/coats',views.coats,name='coats'),
    path('homepage/leatherjacket',views.leatherjacket,name='leatherjacket'),
    path('homepage/corduroypants',views.corduroypants,name='corduroypants'),
    path('homepage/beanies',views.beanies,name='beanies'),
    path('homepage/berets',views.berets,name='berets'),
    path('homepage/cardigans',views.cardigans,name='cardigans'),
    path('homepage/flannelshirts',views.flannelshirts,name='flannelshirts'),
    path('homepage/leggings',views.leggings,name='leggings'),
    path('homepage/suedehandbags',views.suedehandbags,name='suedehandbags'),

    path('homepage/Jacket',views.Jacket,name='Jacket'),
    path('homepage/WoolCoat',views.WoolCoat,name='WoolCoat'),
    path('homepage/CashmereSweater',views.CashmereSweater,name='CashmereSweater'),
    path('homepage/SnowBoots',views.SnowBoots,name='SnowBoots'),
    path('homepage/TurtleneckSweater',views.TurtleneckSweater,name='TurtleneckSweater'),
    path('homepage/DarkColoredJeans',views.DarkColoredJeans,name='DarkColoredJeans'),
    path('homepage/WoolSkirts',views.WoolSkirts,name='WoolSkirts'),
    path('homepage/VelvetDresses',views.VelvetDresses,name='VelvetDresses'),
    path('homepage/OvertheKneeBoots',views.OvertheKneeBoots,name='OvertheKneeBoots'),
    path('homepage/StatementBoots',views.StatementBoots,name='StatementBoots'),
    path('homepage/Gloves',views.Gloves,name='Gloves'),
    path('homepage/upload',views.upload_file,name='upload'),



















]
