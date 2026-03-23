import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

class SentinelEngine:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 3), analyzer='char_wb')
        
        # Define the three experts
        rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        nb = MultinomialNB()

        # Create the Ensemble (Soft Voting uses averaged probabilities)
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr), ('nb', nb)],
            voting='soft'
        )
        self.df_balanced = None
        self.global_df = None

    def prepare_and_train(self, files):
        data_list = []
        for f in files:
            try:
                temp_df = pd.read_csv(f)
                data_list.append(temp_df)
            except: continue

        self.global_df = pd.concat(data_list, ignore_index=True)
        
        # Dynamic Balancing
        df_real = self.global_df[self.global_df.label == 'Real']
        df_fake = self.global_df[self.global_df.label == 'Fake']
        n_samples = min(len(df_real), len(df_fake))
        
        self.df_balanced = pd.concat([
            df_real.sample(n=n_samples, random_state=42),
            df_fake.sample(n=n_samples, random_state=42)
        ])

        X = self.tfidf.fit_transform(self.df_balanced['review_text'])
        y = self.df_balanced['label'].apply(lambda x: 1 if x == 'Fake' else 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        return classification_report(y_test, self.model.predict(X_test), target_names=['Real', 'Fake'])

    def analyze_review(self, text, rating):
        vec = self.tfidf.transform([text])
        prob = self.model.predict_proba(vec)[0][1]  # ML ensemble probability

        # ── Linguistic feature extraction ──────────────────────────────────
        words = text.lower().split()
        word_count = len(words)
        avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        personal_pronouns = sum(1 for w in words if w in ['i', 'me', 'my', 'we', 'us'])

        # Lexical diversity: unique words / total words (AI text tends to be repetitive)
        lexical_diversity = len(set(words)) / word_count if word_count > 0 else 0

        # Punctuation richness: contractions & exclamations signal human casualness
        punctuation_count = sum(1 for ch in text if ch in "!?,;")
        punct_ratio = punctuation_count / word_count if word_count > 0 else 0

        # Generic superlatives AI overuses
        generic_phrases = [
            'highly recommend', 'exceeded my expectations', 'truly exceptional',
            'outstanding service', 'second to none', 'top-notch', 'absolutely amazing',
            'without hesitation', 'five stars', '5 stars', 'best in class',
            'state of the art', 'world class', 'unparalleled', 'seamless experience'
        ]
        generic_hit_count = sum(1 for phrase in generic_phrases if phrase in text.lower())

        slang = ['loot', 'paisa', 'barbad', 'bakwas', 'fraud', 'chor', 'ganda', 'ghatia']
        has_slang = any(word in text.lower() for word in slang)

        sentiment_polarity = TextBlob(text).sentiment.polarity
        sentiment_score = (sentiment_polarity + 1) * 2 + 1
        consistency = 1 - (abs(sentiment_score - rating) / 4)

        # ── Multi-signal AI bot scoring ─────────────────────────────────────
        # Each flag contributes independently; score >= 2 → FAKE, == 1 → SUSPICIOUS
        bot_flags = []

        # Flag 1: ML ensemble says high probability of fake
        if prob > 0.60:
            bot_flags.append(f"🤖 Model confidence: {prob*100:.1f}%")

        # Flag 2: Long, complex words with no personal voice → AI prose signature
        if avg_word_len > 5.5 and personal_pronouns == 0:
            bot_flags.append("📐 Impersonal formal vocabulary")

        # Flag 3: Very low lexical diversity → templated / repetitive generation
        if lexical_diversity < 0.55 and word_count > 20:
            bot_flags.append(f"🔁 Low lexical diversity ({lexical_diversity:.2f})")

        # Flag 4: Minimal punctuation in a long review → AI avoids casual punctuation
        if punct_ratio < 0.03 and word_count > 25:
            bot_flags.append("🔇 Unusually flat punctuation")

        # Flag 5: Multiple generic marketing superlatives
        if generic_hit_count >= 2:
            bot_flags.append(f"📢 {generic_hit_count} generic AI phrases detected")

        # ── Verdict logic ───────────────────────────────────────────────────
        reasons = []

        if consistency < 0.50:
            # Rating-text mismatch is always the primary signal
            verdict, f_type = "🚨 FAKE", "RATING MANIPULATION"
            reasons.append(f"⚖️ Text sentiment({sentiment_score:.1f}) vs Star rating({rating})")

        elif len(bot_flags) >= 2:
            # Two or more independent bot signals → confident FAKE
            verdict, f_type = "🚨 FAKE", "AI BOT SIGNATURE"
            reasons.extend(bot_flags)

        elif len(bot_flags) == 1 or (prob > 0.45 and word_count > 10):
            # One bot signal or borderline ML score → SUSPICIOUS
            verdict, f_type = "⚠️ SUSPICIOUS", "AI BOT SIGNATURE"
            reasons.extend(bot_flags if bot_flags else [f"🤖 Borderline model score: {prob*100:.1f}%"])

        elif word_count < 10 or has_slang:
            verdict, f_type = "⚠️ SUSPICIOUS", "LOW-QUALITY / SLANG"
            if word_count < 10: reasons.append("📝 Too short to analyse reliably")
            if has_slang: reasons.append("🗣️ Hostile / spam slang detected")

        else:
            verdict, f_type = "✅ REAL", "AUTHENTIC"

        # ── Trust score: blend ML + heuristic penalties ─────────────────────
        heuristic_penalty = min(len(bot_flags) * 12, 40)  # up to -40 pts from heuristics
        trust_score = max(0.0, (1 - prob) * 100 - heuristic_penalty)

        return {
            "Verdict": verdict,
            "Type": f_type,
            "Trust": f"{trust_score:.1f}%",
            "FakeConfidence": f"{prob * 100:.1f}%",
            "Explanation": " | ".join(reasons) if reasons else "Organic human feedback."
        }

    def get_business_verdict(self, business_name):
        # FIX 2: Use global_df (full dataset) instead of df_balanced (down-sampled)
        # so no businesses are silently dropped from lookup
        biz_df = self.global_df[self.global_df['business_name'].str.lower() == business_name.lower()]

        # FIX 2b: Return a dict with an "Error" key so the frontend's
        # report.get("Error") check actually fires instead of crashing downstream
        if biz_df.empty:
            return {"Error": f"Business '{business_name}' not found in dataset."}
        
        results = [(row['review_text'], self.analyze_review(row['review_text'], row['rating'])) for _, row in biz_df.iterrows()]
        stats = {"AI BOT SIGNATURE": 0, "RATING MANIPULATION": 0, "LOW-QUALITY / SLANG": 0, "AUTHENTIC": 0}
        for _, res in results:
            stats[res['Type']] += 1
        
        integrity_score = (stats["AUTHENTIC"] / len(biz_df)) * 100
        return {
            "Business": business_name.upper(),
            "Integrity Score": f"{integrity_score:.1f}/100",
            "Recommendation": "🌟 RECOMMENDED" if integrity_score >= 75 else "🚫 NOT RECOMMENDED",
            "Breakdown": stats,
            "Full_Details": results,
            "Analysis": f"Trust Level: {integrity_score:.1f}%"
        }