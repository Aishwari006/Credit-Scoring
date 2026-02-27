from django.contrib import admin
from .models import FinancialUpload, RawTransaction, CompanyCreditProfile

admin.site.register(FinancialUpload)
admin.site.register(RawTransaction)
admin.site.register(CompanyCreditProfile)