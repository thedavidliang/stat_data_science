import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class UserSegmentationEngine:
    '''
    Production-ready user segmentation and personalization system

    Features:
        - behavioral clustering (k-means)
        - churn prediction models
        - conversion propensity scoring
        - feature-based recommendations
        - targeted intervention triggers
    '''

    def __init__(self):
        self.scalar = StandardScaler()
        self.cluster_model = None
        self.churn_model = None
        self.conversion_model = None
        self.cluster_features = None
        self.churn_features = None
        self.conversion_features = None
        self.segment_profiles = {}

   
    def create_user_segments(
        self,
        df: pd.DataFrame,  # restrict to feature columns
        features: list[str] = None,
        n_clusters: int = 5,
        
    ) -> pd.DataFrame:
        '''
        Segment users into behavior cohorts using K-means

        arguments:
            - df: dataframe of user information
            - features: list of features in data to use for clustering 
                (sessions_last_30d, days_since_last_visit, avg_session_duration, avg_session_purchase_probability, avg_session_purchase_amount, email_open_rate, feature_adoption_rate,)
            - n_clusters: number of segments to create
        
        returns:
            dataframe with segment assignments
        '''
        if features is not None:
            X = df[features].copy()
            self.cluster_features = features
        else:
            X = df.copy()
            self.cluster_features = X.columns
        X_scaled = self.scalar.fit_transform(X)   # maybe move this to a separate data-ingestion step?
        np.nan_to_num(X_scaled, copy=False, nan=0)
        # X_scaled = pd.DataFrame(X_scaled).fillna(0).values # (alternative)

        # fit k-means
        self.cluster_model = KMeans(n_clusters = n_clusters, random_state = 13, n_init = 10)
        df['segment'] = self.cluster_model.fit_predict(X_scaled)

        # calculate silhouette score
        silhouette = silhouette_score(X_scaled, df['segment'])
        print(f'Silhouette score: {silhouette:0.3f}')

        # profile segments:
        self._profile_segments(df, features)

        return df


    def _profile_segments(self, 
        df: pd.DataFrame, 
        features: list[str],
    ) -> None:
        '''
        Create profile for each segment of the data

        '''
        # profile of each segment:
        # name, size, size_pct, avg_sessions, avg_spend, churn_rate, conversion_rate, characteristics (another dictionary. something for each feature)
        segments = df.segment.unique()
        for s in segments:
            df_segment = df[df.segment == s]
            profile = {
                'name': s,
                'size': len(df_segment),
                'size_pct': len(df_segment) / len(df),
                'avg_sessions_30d': df_segment.sessions_last_30d.mean(),
                'avg_spend': df_segment.total_spend.mean(),
                'avg_churn_rate': df_segment.churned.mean(),
                'avg_conversion_rate': df_segment.converted.mean(),
                'characteristics': {},   # summaries for each feature columns
            }
                
            for f in features:
                pop_mean = df[f].mean()
                segment_mean = df_segment[f].mean()
                profile['characteristics'][f] = {
                    'mean': segment_mean,
                    'deviation': (segment_mean - pop_mean) / pop_mean,
                }
            self.segment_profiles[s] = profile
        
        self._print_segment_profiles()

    
    def _print_segment_profiles(self):
        '''
        Print self.segment_profiles
        '''
        print('='*80)
        print('Segment Profiles')
        print('-'*80)
        for segment, profile in self.segment_profiles.items():
            print(f'* Segment: {segment}')
            for k,v in profile.items():
                if k!= 'characteristics':
                    print(f'    {k}: {v}')
                else:
                    print(f'    {k}:')
                    for ck, cv in v.items():
                        print(f'      {ck}: {cv}')

    def predict_user_segments(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        
        df = df.copy()
        X = df[self.cluster_features]
        X_scaled = self.scalar.transform(X)
        np.nan_to_num(X_scaled, copy=False, nan=0)

        df['segment'] = self.cluster_model.predict(X_scaled)

        silhouette = silhouette_score(X_scaled, df['segment'])
        print(f'Silhouette score: {silhouette:0.3f}')

        return df

    def train_model(self,
        df: pd.DataFrame,
        model,
        target: str,
        features: list[str] = None,
    ) -> dict:
        '''
        Train the appropriate model, saves model, reports performance metrics

        Arguments:
            - df: pandas DataFrame containing user info
            - features: features in df to use in model
            - model: model to fit (eg. GradientBoostingClassifier(), or custome pipeline including a scaler)
            - target: either {'churned' or 'converted'}
        Returns:
            - performance metrics of model
        '''
        
        assert target in ['churned','converted'], "'target' must be in ['churned','converted']"

        # get train/test data
        if features is None:
            X = df.drop(columns=target).fillna(0).copy()
            features = X.columns
        else:
            X = df[features].fillna(0).copy()
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=33, stratify=y)

        # fit model
        model.fit(X_train, y_train)

        # evaluate model
        y_test_hat = model.predict(X_test)
        y_test_hat_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_test_hat_prob)

        # feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # print 
        print('='*80)
        print(f'Prediction Model: {target}')
        print('-'*80)
        print(f'ROC-AUC Score: {auc}')
        print(classification_report(y_test, y_test_hat, target_names = ['not '+target, target]))
        print(f'\nTop {target} Predictors:')
        for _,row in feature_importance.iterrows():
            print(f' *  {row.feature}:  {row.importance:.3f}')

        # save model
        if target=='churned':
            self.churn_model = model
            self.churn_features = features
        else:
            self.conversion_model = model
            self.conversion_features = features

        return {
            'auc': auc,
            'feature_importance': feature_importance
        }

    def score_users(self,
        df: pd.DataFrame,
    ):
        '''
        predict churn and conversion scores. put into bins
        '''
        df = df.copy()
        
        # churn likelihood
        X_churn = df[self.churn_features].fillna(0)
        df['churn_likelihood'] = self.churn_model.predict_proba(X_churn)[:,1]

        # conversion likelihood
        X_conversion = df[self.conversion_features].fillna(0)
        df['conversion_likelihood'] = self.conversion_model.predict_proba(X_conversion)[:,1]

        df['churn_risk_category'] = pd.cut(
            df.churn_likelihood,
            bins=[0, 0.3, 0.6, 1],
            labels=['low','medium','high'],
        )
        df['conversion_likelihood_category'] = pd.cut(
            df.conversion_likelihood,
            bins=[0, 0.3, 0.6, 1],
            labels=['low','medium','high'],
        )
        return df
        

    def recommend_intervention(self, 
        df: pd.DataFrame,
        n_recommendations: int=10,
    ) -> pd.DataFrame:
        '''
        Recommend interventions for high-value users at risk

        Arguments:
            - df: user data with scores (from self.score_users)
            - n_recommendations: number of users to recommend interventions for

        Returns:
            - dataframe with intervention recommendations
        '''

        df = df.copy()

        interventions = {
            'high_churn_high_conversion':{
                'priority': 1,
                'action': 'Personal outreach from accounts manager',
                'message': 'Schedule 1-on-1 demo meeting',
            },
            'high_churn_medium_conversion':{
                'priority': 2,
                'action': 'Targetted discount offer',
                'message': 'Limited discount on subscription upgrade',
            },
            'high_churn_low_conversion':{
                'priority': 3,
                'action': 'Targetted re-engagement email campaign',
                'message': 'Testimonials',
            },
            'medium_churn_high_conversion':{
                'priority': 2,
                'action': 'Feature recommendation',
                'message': 'Showcase how premium features align with usage',
            },
        }
        # get intervention score to determine intervention priority
        df['intervention_score'] = (
            0.5 * df.churn_likelihood + 
            0.15 * df.total_spend / (df.total_spend.max()) + 
            0.35 * df.conversion_likelihood
        )

        users_at_risk = df.nlargest(n_recommendations, 'intervention_score')

        recommendations = []
        for _, row in users_at_risk.iterrows():
            intervention_type = f'{row.churn_risk_category}_churn_{row.conversion_likelihood_category}_conversion'
            intervention = interventions[intervention_type]
            segment_name = self.segment_profiles[row.segment]['name']
            recommendations.append({
                'user_id': row.user_id,
                'segment': segment_name,
                'churn_likelihood': round(row.churn_likelihood, 3),
                'conversion_likelihood': round(row.conversion_likelihood, 3),
                'priority': intervention['priority'],
                'intervention': intervention['action'],
                'message': intervention['message'],
                'estimated_ltv': row.total_spend * 2,
            })
        df_recommendations = pd.DataFrame(recommendations)
        return df_recommendations
    
    def visualize_segments(self, df):
        """Create visualizations of user segments."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Segment sizes
        segment_counts = df['segment'].value_counts().sort_index()
        segment_names = [self.segment_profiles[i]['name'] for i in segment_counts.index]
        axes[0, 0].bar(range(len(segment_counts)), segment_counts.values, color=plt.cm.Set3(range(len(segment_counts))))
        axes[0, 0].set_xticks(range(len(segment_counts)))
        axes[0, 0].set_xticklabels(segment_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Number of Users')
        axes[0, 0].set_title('Segment Distribution', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # 2. Churn rate by segment
        churn_by_segment = df.groupby('segment')['churned'].mean()
        axes[0, 1].bar(range(len(churn_by_segment)), churn_by_segment.values * 100, color=plt.cm.Set3(range(len(churn_by_segment))))
        axes[0, 1].set_xticks(range(len(churn_by_segment)))
        axes[0, 1].set_xticklabels(segment_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Churn Rate (%)')
        axes[0, 1].set_title('Churn Rate by Segment', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # 3. Conversion rate by segment
        conversion_by_segment = df.groupby('segment')['converted'].mean()
        axes[1, 0].bar(range(len(conversion_by_segment)), conversion_by_segment.values * 100, color=plt.cm.Set3(range(len(conversion_by_segment))))
        axes[1, 0].set_xticks(range(len(conversion_by_segment)))
        axes[1, 0].set_xticklabels(segment_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Conversion Rate (%)')
        axes[1, 0].set_title('Conversion Rate by Segment', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Risk score distribution
        axes[1, 1].hist(df['churn_likelihood'], bins=30, alpha=0.4, color='red', label='Churn Risk', edgecolor='black')
        axes[1, 1].hist(df['conversion_likelihood'], bins=30, alpha=0.4, color='green', label='Conversion Propensity', edgecolor='black')
        axes[1, 1].set_xlabel('Score')
        axes[1, 1].set_ylabel('Number of Users')
        axes[1, 1].set_title('Risk Score Distribution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        