"""
Advanced Analysis Functions for Shark Tank Data
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedSharkTankAnalyzer:
    """Advanced analysis using machine learning"""
    
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_importance = {}
    
    def prepare_ml_data(self, datasets):
        """Prepare data for machine learning"""
        all_data = []
        
        # Handle both serialized and DataFrame formats
        if isinstance(datasets, dict) and "data" in list(datasets.values())[0]:
            # Convert serialized datasets back to DataFrames
            for country, data_info in datasets.items():
                df = pd.DataFrame(data_info["data"])
                all_data.append(self._process_dataframe(df, country))
        else:
            # Handle regular DataFrames
            for country, df in datasets.items():
                all_data.append(self._process_dataframe(df, country))
        
        if not all_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Clean data
        combined_df = combined_df.dropna(subset=['ask_amount', 'equity_offered', 'got_deal'])
        
        # Encode categorical variables
        if 'industry' in combined_df.columns:
            le_industry = LabelEncoder()
            combined_df['industry_encoded'] = le_industry.fit_transform(combined_df['industry'])
            self.encoders['industry'] = le_industry
        
        if 'gender' in combined_df.columns:
            le_gender = LabelEncoder()
            combined_df['gender_encoded'] = le_gender.fit_transform(combined_df['gender'])
            self.encoders['gender'] = le_gender
        
        if 'country_encoded' in combined_df.columns:
            le_country = LabelEncoder()
            combined_df['country_encoded'] = le_country.fit_transform(combined_df['country_encoded'])
            self.encoders['country'] = le_country
        
        return combined_df
    
    def _process_dataframe(self, df, country):
        """Process a single dataframe for ML"""
        if df.empty:
            return df
            
        # Create a copy for processing
        data = df.copy()
        
        # Add country as feature
        data['country_encoded'] = country
        
        # Ensure we have the required columns
        required_cols = ['ask_amount', 'equity_offered', 'got_deal', 'industry']
        if all(col in data.columns for col in required_cols):
            return data[required_cols + ['country_encoded', 'gender'] if 'gender' in data.columns else required_cols + ['country_encoded']]
        else:
            return pd.DataFrame()
    
    def train_success_predictor(self, datasets):
        """Train ML model to predict deal success"""
        df = self.prepare_ml_data(datasets)
        
        if df.empty or 'got_deal' not in df.columns:
            return {"error": "Insufficient data for training"}
        
        # Prepare features
        feature_cols = ['ask_amount', 'equity_offered']
        if 'industry_encoded' in df.columns:
            feature_cols.append('industry_encoded')
        if 'gender_encoded' in df.columns:
            feature_cols.append('gender_encoded')
        if 'country_encoded' in df.columns:
            feature_cols.append('country_encoded')
        
        X = df[feature_cols]
        y = df['got_deal']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store model and feature importance
        self.models['success_predictor'] = model
        self.feature_importance['success_predictor'] = dict(zip(feature_cols, model.feature_importances_))
        
        return {
            "accuracy": accuracy,
            "feature_importance": self.feature_importance['success_predictor'],
            "classification_report": classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_pitch_success(self, pitch_data):
        """Predict success for a specific pitch"""
        if 'success_predictor' not in self.models:
            return {"error": "Model not trained"}
        
        model = self.models['success_predictor']
        
        # Prepare features
        features = []
        
        # Ask amount
        ask_amount = pitch_data.get('ask_amount', 0)
        features.append(ask_amount)
        
        # Equity offered
        equity_offered = pitch_data.get('equity_offered', 0)
        features.append(equity_offered)
        
        # Industry (if available)
        if 'industry' in pitch_data and 'industry' in self.encoders:
            try:
                industry_encoded = self.encoders['industry'].transform([pitch_data['industry']])[0]
                features.append(industry_encoded)
            except:
                features.append(0)  # Default value
        else:
            features.append(0)
        
        # Gender (if available)
        if 'gender' in pitch_data and 'gender' in self.encoders:
            try:
                gender_encoded = self.encoders['gender'].transform([pitch_data['gender']])[0]
                features.append(gender_encoded)
            except:
                features.append(0)
        else:
            features.append(0)
        
        # Country (if available)
        if 'country' in pitch_data and 'country' in self.encoders:
            try:
                country_encoded = self.encoders['country'].transform([pitch_data['country']])[0]
                features.append(country_encoded)
            except:
                features.append(0)
        else:
            features.append(0)
        
        # Make prediction
        prediction = model.predict([features])[0]
        probability = model.predict_proba([features])[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),  # Probability of success
            "confidence": float(max(probability))
        }
    
    def analyze_investment_patterns(self, datasets):
        """Analyze investment patterns across sharks and countries"""
        patterns = {}
        
        # Handle both serialized and DataFrame formats
        if isinstance(datasets, dict) and "data" in list(datasets.values())[0]:
            # Convert serialized datasets back to DataFrames
            for country, data_info in datasets.items():
                df = pd.DataFrame(data_info["data"])
                patterns[country] = self._analyze_country_patterns(df, country)
        else:
            # Handle regular DataFrames
            for country, df in datasets.items():
                patterns[country] = self._analyze_country_patterns(df, country)
        
        return patterns
    
    def _analyze_country_patterns(self, df, country):
        """Analyze patterns for a single country"""
        if country == "merged" or df.empty:
            return {}
        
        country_patterns = {
            "total_pitches": len(df),
            "success_rate": df['got_deal'].mean() if 'got_deal' in df.columns else 0,
            "avg_ask_amount": df['ask_amount'].mean() if 'ask_amount' in df.columns else 0,
            "avg_equity_offered": df['equity_offered'].mean() if 'equity_offered' in df.columns else 0
        }
        
        # Industry analysis
        if 'industry' in df.columns and 'got_deal' in df.columns:
            industry_success = df.groupby('industry')['got_deal'].agg(['count', 'sum', 'mean']).round(3)
            industry_success.columns = ['total_pitches', 'successful_deals', 'success_rate']
            country_patterns['industry_analysis'] = industry_success.to_dict('index')
        
        # Gender analysis
        if 'gender' in df.columns and 'got_deal' in df.columns:
            gender_success = df.groupby('gender')['got_deal'].agg(['count', 'sum', 'mean']).round(3)
            gender_success.columns = ['total_pitches', 'successful_deals', 'success_rate']
            country_patterns['gender_analysis'] = gender_success.to_dict('index')
        
        return country_patterns
    
    def generate_insights(self, datasets):
        """Generate actionable insights from the data"""
        insights = []
        
        # Handle both serialized and DataFrame formats
        all_data = []
        if isinstance(datasets, dict) and "data" in list(datasets.values())[0]:
            # Convert serialized datasets back to DataFrames
            for country, data_info in datasets.items():
                if country == "merged":
                    continue
                df = pd.DataFrame(data_info["data"])
                if 'got_deal' in df.columns:
                    df_copy = df.copy()
                    df_copy['country'] = country
                    all_data.append(df_copy)
        else:
            # Handle regular DataFrames
            for country, df in datasets.items():
                if country == "merged" or df.empty:
                    continue
                if 'got_deal' in df.columns:
                    df_copy = df.copy()
                    df_copy['country'] = country
                    all_data.append(df_copy)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Success rate by industry
            if 'industry' in combined_df.columns and 'got_deal' in combined_df.columns:
                industry_success = combined_df.groupby('industry')['got_deal'].mean().sort_values(ascending=False)
                top_industry = industry_success.index[0]
                top_rate = industry_success.iloc[0]
                insights.append(f"Most successful industry: {top_industry} ({top_rate:.1%} success rate)")
            
            # Optimal equity range
            if 'equity_offered' in combined_df.columns and 'got_deal' in combined_df.columns:
                successful_deals = combined_df[combined_df['got_deal'] == 1]
                if not successful_deals.empty:
                    avg_equity = successful_deals['equity_offered'].mean()
                    insights.append(f"Optimal equity offer: {avg_equity:.1f}% (average for successful deals)")
            
            # Ask amount analysis
            if 'ask_amount' in combined_df.columns and 'got_deal' in combined_df.columns:
                successful_deals = combined_df[combined_df['got_deal'] == 1]
                if not successful_deals.empty:
                    median_ask = successful_deals['ask_amount'].median()
                    insights.append(f"Median ask amount for successful deals: ${median_ask:,.0f}")
        
        return insights
