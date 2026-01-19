import pytest
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

def test_smote_pipeline_integration():
    # Create synthetic imbalanced data
    # Class 0: 20 samples, Class 1: 5 samples
    texts = ["bagus sekali"] * 20 + ["jelek parah"] * 5
    labels = [0] * 20 + [1] * 5
    
    # Simple TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Pipeline
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42, k_neighbors=1)), # k=1 because only 5 samples
        ('svm', LinearSVC(random_state=42))
    ])
    
    # Fit
    try:
        pipeline.fit(X, labels)
    except Exception as e:
        pytest.fail(f"Pipeline fitting failed: {e}")
        
    # Predict
    preds = pipeline.predict(X)
    assert len(preds) == 25

def test_smote_resampling_effect():
    # Verify SMOTE actually increases samples
    texts = ["bagus"] * 20 + ["jelek"] * 5
    labels = [0] * 20 + [1] * 5
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    smote = SMOTE(random_state=42, k_neighbors=1)
    X_res, y_res = smote.fit_resample(X, labels)
    
    # Should be balanced (20 vs 20)
    assert len(y_res) == 40
    assert sum(y_res) == 20
