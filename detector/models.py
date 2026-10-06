from django.db import models


class TransactionCheck(models.Model):
    """Stores a log of transactions submitted for fraud prediction."""

    amount = models.FloatField()
    fraud_probability = models.FloatField()
    is_fraud = models.BooleanField()
    checked_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Txn ₹{self.amount} - {'FRAUD' if self.is_fraud else 'OK'} ({self.fraud_probability:.2%})"

    class Meta:
        ordering = ["-checked_at"]
