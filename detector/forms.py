from django import forms


class TransactionForm(forms.Form):
    """
    Simplified real-world input: the amount, plus a 'risk profile' the user
    picks, which we map to a synthetic 28-feature vector (V1..V28) matching
    what the ANN was trained on. This mirrors how a real integration would
    receive pre-engineered/anonymized features from an upstream system.
    """

    RISK_CHOICES = [
        ("normal", "Typical transaction"),
        ("unusual", "Unusual pattern (e.g. new location/device)"),
        ("high_risk", "High-risk pattern (e.g. rapid repeated charges)"),
    ]

    amount = forms.FloatField(
        min_value=0,
        label="Transaction Amount (₹)",
        widget=forms.NumberInput(attrs={"class": "form-control", "placeholder": "e.g. 4500"}),
    )
    risk_profile = forms.ChoiceField(
        choices=RISK_CHOICES,
        label="Transaction Pattern",
        widget=forms.Select(attrs={"class": "form-select"}),
    )
