from django.db import models
from django.db.models import JSONField

# 1. The KYC Model (This saves the form data)
class FinancialUpload(models.Model):
    business_name = models.CharField(max_length=200)
    gstin = models.CharField(max_length=15)
    pan = models.CharField(max_length=10)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.business_name

# 2. The Time-Series Model (This saves the CSV rows for the charts)
class RawTransaction(models.Model):
    nameOrig = models.CharField(max_length=100, db_index=True)
    date = models.DateTimeField()
    txn_type = models.CharField(max_length=50)
    amount = models.FloatField()
    nameDest = models.CharField(max_length=100, blank=True, null=True)
    balance = models.FloatField()

    def __str__(self):
        return f"{self.nameOrig} | {self.date.strftime('%Y-%m-%d')} | {self.txn_type} | ₹{self.amount}"
 
 #3.   
class CompanyCreditProfile(models.Model):
    business_name = models.CharField(max_length=200, unique=True)
    ml_features = JSONField(null=True, blank=True) 
    
    # Core Outputs
    credit_score = models.IntegerField(null=True, blank=True)
    decision = models.CharField(max_length=50, null=True, blank=True)
    
    # --- NEW EXPLAINABLE AI FIELDS ---
    probability_of_default = models.FloatField(null=True, blank=True)
    decision_threshold = models.FloatField(null=True, blank=True)
    risk_bucket = models.CharField(max_length=100, null=True, blank=True)
    top_risk_drivers = JSONField(null=True, blank=True) # Stores the list of reasons
    # ---------------------------------
    
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.business_name} - Score: {self.credit_score}"