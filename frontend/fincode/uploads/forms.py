from django import forms
from .models import FinancialUpload

class FinancialUploadForm(forms.ModelForm):
    class Meta:
        model = FinancialUpload
        fields = ['business_name', 'gstin', 'pan', 'file']
        widgets = {
            'business_name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'Enter your business name'
            }),
            'gstin': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '22AAAAA0000A1Z5'
            }),
            'pan': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': 'ABCDE1234F'
            }),
            'file': forms.ClearableFileInput(attrs={
                'class': 'form-control',
            })
        }

    def clean_file(self):
        file = self.cleaned_data['file']
        if not file.name.endswith(('.csv', '.xlsx', '.xls')):
            raise forms.ValidationError("Only CSV or Excel files allowed.")
        return file