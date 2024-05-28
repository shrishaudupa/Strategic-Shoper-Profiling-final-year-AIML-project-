from django import forms
from .models import Product

class ProductForm(forms.ModelForm):
    class Meta:
        model = Product
        fields = ['name','price']
class QuantityForm(forms.Form):
    number= forms.IntegerField()
    product_id=forms.IntegerField()
class CameraForm(forms.Form):
    asc=forms.IntegerField()

class UploadFileForm(forms.Form):
    file = forms.FileField(label='Select a CSV file', help_text='Max. 5MB')




