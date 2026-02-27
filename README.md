# sms-spam-classifier-
# DistilBERT SMS Spam Classifier (PyTorch)

Ένα mini‑project που υλοποιεί ταξινόμηση SMS μηνυμάτων (spam vs ham) χρησιμοποιώντας το DistilBERT.  
Το project περιλαμβάνει πλήρη ροή: φόρτωση δεδομένων, tokenization, fine‑tuning, αξιολόγηση, αποθήκευση μοντέλου και prediction.

---

## Χαρακτηριστικά
- Χρήση DistilBERT (distilbert-base-uncased)
- PyTorch training loop
- 97–99% accuracy στο SMS Spam dataset
- Prediction function για νέα μηνύματα
- Αποθήκευση & φόρτωση μοντέλου
- Καθαρός, απλός κώδικας για εκμάθηση NLP με Transformers

---

## Δομή Project
sms-spam-classifier/ │ ├── model/ │   ├── clf/ │   ├── info.pkl │ ├── spam_classifier.py ├── SMSSpamCollection ├── README.md ├── requirements.txt └── LICENSE

---

## Εκτέλεση

1. Κατέβασε το dataset από το UCI ML Repository.
2. Βάλε το αρχείο `SMSSpamCollection` στο ίδιο directory.
3. Εγκατέστησε τα requirements:

```bash
pip install -r requirements.txt
```
4. Τρέξε το script:
python spam_classifier.py

# Prediction Example
predict("Congratulations! You won a free ticket!")
# → "spam"

# Αποτελέσματα
Το DistilBERT πετυχαίνει:
- 97–99% accuracy
- Πολύ καλή γενίκευση σε πραγματικά SMS

