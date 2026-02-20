
import numpy as np # type: ignore
import pandas as pd # type: ignore
from datetime import datetime, timedelta
from collections import defaultdict

class UserDataGenerator:

    def __init__(
            self, 
            n_users: int,
            time_back: int = 100,
            signup_time_distribution = np.random.uniform,
            random_seed = 2,
            **kwargs,
        ):
        '''
        Arguments:
            - n_users:  number of users to generate data for
            - time_back:  number of days across which users can have joined 
            - signup_time_distribution:  distribution of user signup_time between today and (today-time_back)
            - random_seed:  seed for numpy.random
        '''
        # user initialization data
        self.n_users = n_users
        self.time_back = time_back
        self.signup_time_distribution = signup_time_distribution
        self.random_seed = random_seed
        self.today = datetime.today()

        # dataframes of user data
        self.users = None
        self.sessions = None
        self.features = None
        self.emails = None
        self.support_tickets = None
        self.referrals = None 

        self.user_summary = None

        self._generate_user_profiles(signup_time_distribution, **kwargs)

    # generate df with: users, signup_datetime (according to self.time_back)
    def _generate_user_profiles(self, signup_time_distribution, **kwargs) -> pd.DataFrame:
        '''
        Generate set of users and their times of signup

        Arguments:
            - signup_time_distribution:  distribution of user times since signup. (eg. np.random.beta)
            - kwargs:  relevant parameters for signup_time_distribution (eg. a,b for np.random.beta)
        '''
        np.random.seed(self.random_seed)
        user_data = {
            'user_id': [f'user_{i}' for i in range(self.n_users)],
            'signup_datetime': [self.today - timedelta(days=signup_time_distribution(**kwargs)*self.time_back) for i in range(self.n_users)],
        }
        self.users = pd.DataFrame(user_data)
        return self.users

    def _generate_sessions(
            self,
            session_rates: list[float] = None,   # eg. np.arange(self.n_user)%3+1
            session_duration_rates: list[float] = None,
            purchase_probabilities: list[float] = None,
            purchase_means: list[float] = None,
            mobile_probabilities: list[float] = None,
            random_seed: int = None,
        ) -> pd.DataFrame:
        '''
        Generate sessions table
        
        Arguments:
            - session_rates: list of len self.n_users, representing avg time (in days) between sessions for each user (scale parameter for exponential distribution)
            - session_duration_rates: list of len self.n_users , representing avg duration (in days) of session for each user (scale parameter for exponential distribution)
            - purchase_probabilities: list of len self.n_users, representing probability each user makes a purchase during session
            - purchase_means: list of len self.n_users, representing mean purchase amount of user, if they make a purchase

        Returns:
            - pd.DataFrame with attributes [user_id, session_id, start_datetime, end_datetime, purchase_amount]
        '''
        assert self.users is not None, 'need to generate users table first'
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if session_rates is None:
            session_rates = [10]*self.n_users
        if session_duration_rates is None:
            session_duration_rates = [0.05]*self.n_users
        if purchase_probabilities is None:
            purchase_probabilities = [0.25]*self.n_users
        if purchase_means is None:
            purchase_means = [10]*self.n_users
        if mobile_probabilities is None:
            mobile_probabilities = [0.5]*self.n_users

        sessions_data = defaultdict(list)
        session_counter = 0
        for (_,row),sr, sdr, pp, pm, mp in zip(self.users.iterrows(), session_rates, session_duration_rates, purchase_probabilities, purchase_means, mobile_probabilities):
            start = row.signup_datetime
            while start < self.today:
                end = start + timedelta(days = np.random.exponential(sdr))
                sessions_data['user_id'].append(row.user_id)
                sessions_data['session_id'].append(session_counter)
                sessions_data['start_datetime'].append(start)
                sessions_data['end_datetime'].append(end)
                if np.random.uniform() < pp:
                    sessions_data['purchase_amount'].append(int(np.random.chisquare(pm)))
                else:
                    sessions_data['purchase_amount'].append(0)
                sessions_data['device'] = 'mobile' if np.random.uniform() < mp else 'desktop'
                start = end + timedelta(days=np.random.exponential(sr))
                session_counter += 1
        self.sessions = pd.DataFrame(sessions_data)
        return self.sessions

    def _generate_product_features(
            self,
            n_features: int = 40,
            feature_release_distribution = np.random.uniform,
            feature_adoption_probabilities: list[float] = None,
            random_seed: int = None,
            **kwargs
        ) -> pd.DataFrame:
        '''
        Generate dataframe of features and whether users adopted them
        
        Arguments:
            - n_features:  number of features
            - feature_release_distribution:  distribution of times since feature releases
            - feature_adoption_probabilies: list of len self.n_users of their probabilities of adopting a new feature
            - kwargs: relevant parameters for feature_release_distribution
        
        Returns:
            - pd.DataFrame with attributes [user_id, feature_id, feature_release_date, adopted]
        '''
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if feature_adoption_probabilities is None:
            feature_adoption_probabilities = np.random.beta(a=np.arange(self.n_users)%5+3, b=5)

        features_data = defaultdict(list)
        start = self.users.signup_datetime.min()
        for i in range(n_features):
            release_date = start + (self.today - start) * feature_release_distribution(**kwargs)
            df_signed_up = self.users[self.users.signup_datetime < release_date]
            for j, row in df_signed_up.iterrows():
                features_data['user_id'].append(row.user_id)
                features_data['feature_id'].append(f'feature_{i}')
                features_data['feature_release_date'].append(release_date)
                features_data['adopted'].append(np.random.uniform() < feature_adoption_probabilities[j])
        self.features = pd.DataFrame(features_data)
        return self.features

    def _generate_emails(
            self,
            email_open_probabilities: list[float] = None,
            email_frequency=7,
            random_seed: int = None,
        ) -> pd.DataFrame:
        '''
        Created dataframe of emails sent to users and whether emails were opened, save into self.emails
        
        Arguments:
            - email_open_proabilities:  list of len self.n_users, corresponding to probability that each is opened
            - email_frequency:  how many days between emails
        '''
        if random_seed is not None:
            np.random.seed(random_seed)
            
        if email_open_probabilities is None:
            email_open_probabilities = np.random.beta(a = np.arange(self.n_users)%4+2, b = 3)

        email_data = defaultdict(list)
        start = self.users.signup_datetime.min() + timedelta(days=2)
        while start < self.today:
            df_signed_up = self.users[self.users.signup_datetime < start]
            for i, row in df_signed_up.iterrows():
                email_data['user_id'].append(row.user_id)
                email_data['email_id'].append(f'email_{start.date()}_{row.user_id}')
                email_data['email_date'].append(start.date())
                email_data['opened'].append(np.random.uniform() < email_open_probabilities[i])
            start += timedelta(days=email_frequency)
        self.emails = pd.DataFrame(email_data)
        return self.emails

    def _generate_support_tickets(
            self,
            support_ticket_rates: list[float] = None,
            support_ticket_resolution_rates: list[float] = None,
            random_seed = None,
        ) -> pd.DataFrame:
        '''
        Generate sessions table
        
        Arguments:
            - support_ticket_rates: list of len self.n_users, representing avg time (in days) between support tickets for each user (scale parameter for exponential distribution)
            - support_ticket_resolution_rate: list of len self.n_users , representing avg duration (in days) of support ticket for each user (scale parameter for exponential distribution)
            - random_seed:  np.random seed
        Returns:
            - pd.DataFrame with attributes [user_id, support_ticket_id, open_datetime, close_datetime]
        '''
        if random_seed is not None:
            np.random.seed(random_seed)

        if support_ticket_rates is None:
            support_ticket_rates = [50]*self.n_users
        if support_ticket_resolution_rates is None:
            support_ticket_resolution_rates = [0.2]*self.n_users
        support_ticket_data = defaultdict(list)
        ticket_counter = 0
        for (_,row), tr, trr in zip(self.users.iterrows(), support_ticket_rates, support_ticket_resolution_rates):
            start = row.signup_datetime + timedelta(days = np.random.exponential(tr))
            while start < self.today:
                end = start + timedelta(days = np.random.exponential(trr))
                support_ticket_data['user_id'].append(row.user_id)
                support_ticket_data['support_ticket_id'].append(f'ticket_{ticket_counter}')
                support_ticket_data['open_datetime'].append(start)
                support_ticket_data['close_datetime'].append(end)
                start = end + timedelta(days = np.random.exponential(tr))
            ticket_counter += 1
        self.support_tickets = pd.DataFrame(support_ticket_data)
        return self.support_tickets
        
    def _generate_referrals(
            self,
            n_referrals: int = 100,
            referral_rates: list[float] = None,
            random_seed: int = None,
        ) -> pd.DataFrame:
        '''
        Generate table of user referrals.  Ensure that referred user is always newer
        
        Arguments:
            - n_referrals:  number of referral pairings to generate
            - referral_rates:  relative weight of likelihoods of users to be in a referral pairing
            - random_seed:  np.random seed
            
        Regurns: pd.DataFrame of referral pairings
        '''
        if random_seed is not None:
            np.random.seed(random_seed)

        if referral_rates is None:
            referral_rates = np.ones(self.n_users)
        referral_rates = np.array(referral_rates)  # cast to np.ndarray
        if np.isclose(referral_rates.sum(), 0):
            referral_rates = np.ones(self.n_users)
        referral_rates = referral_rates / referral_rates.sum()

        referrals_data = defaultdict(list)
        for i in range(n_referrals):
            x,y = np.random.choice(np.arange(self.n_users), p=referral_rates, size=2, replace=False)
            x_user, y_user = f'user_{x}', f'user_{y}'
            x_signup = self.users[self.users.user_id==x_user].signup_datetime.item()
            y_signup = self.users[self.users.user_id==y_user].signup_datetime.item()
            if x_signup<y_signup:
                referrals_data['user_id'].append(x_user)
                referrals_data['referred_user_id'].append(y_user)
            else:
                referrals_data['user_id'].append(y_user)
                referrals_data['referred_user_id'].append(x_user)
        self.referrals = pd.DataFrame(referrals_data)
        return self.referrals

    # generate all data tables
    def generate_all_data(
            self,
            args_sessions: dict = {},
            args_features: dict = {},
            args_emails: dict = {},
            args_support_tickets: dict = {},
            args_referrals: dict = {},
        ) -> tuple[pd.DataFrame]:
        '''
        Generates all relevant sample data
        
        Arguments:
            - args_[x]: dictionary of parameters to be passed to 'self._generate_[x]' function

        Returns: tuple of pd.DataFrames with randomly-generated data
        '''
        return (
            self._generate_sessions(**args_sessions),
            self._generate_product_features(**args_features),
            self._generate_emails(**args_emails),
            self._generate_support_tickets(**args_support_tickets),
            self._generate_referrals(**args_referrals),
        )
        
    # summarize user,sessions,features,emails,tickets,referrals
    def get_user_data_statistics(self) -> pd.DataFrame:
        '''
        Summarize user behavior into single dataframe
        '''
        assert (self.sessions is not None) and (self.features is not None) and (self.emails is not None) and (self.support_tickets is not None) and (self.referrals is not None), 'need to generate data first with generate_all_date()'
        df = self.users.copy()

        # ** Features from users **
        # days_since_signup
        df['days_since_signup'] = self.users.signup_datetime.apply(lambda x: (self.today-x).days)
        df.drop(labels='signup_datetime', axis=1, inplace=True)

        # ** Features from sessions ** 
        sessions = self.sessions.copy()
        
        # sessions_last_30d
        df_sess_last_30d = sessions.loc[sessions.start_datetime > self.today - timedelta(days=30),['user_id','device']].groupby('user_id').agg('count').reset_index()
        df_sess_last_30d.columns = ['user_id','sessions_last_30d']
        df = df.merge(df_sess_last_30d, how='left', on='user_id').fillna(0)
        
        # days_since_last_visit
        df_last_visit = sessions[['user_id','start_datetime']].groupby('user_id').agg('max').reset_index()
        df_last_visit['days_since_last_visit'] = df_last_visit.start_datetime.apply(lambda x: (self.today-x).days)
        df = df.merge(df_last_visit[['user_id','days_since_last_visit']], how='left', on='user_id') 
        df.loc[df.days_since_last_visit.isna(), 'days_since_last_visit'] = df[df.days_since_last_visit.isna()].days_since_signup
        
        # avg_session_duration_sec, avg_session_purchase_amount, avg_session_purchase_probability, total_spend, mobile_usage_pct, weekend_usage_pct
        sessions['duration'] = sessions.end_datetime - sessions.start_datetime
        sessions['duration_sec'] = sessions.duration.apply(lambda x: int(x.days*24*60*60 + x.seconds))
        sessions['purchase_made'] = sessions.purchase_amount > 0
        sessions['is_mobile'] = sessions.device == 'mobile'
        sessions['is_weekend'] = sessions.start_datetime.apply(lambda x: x.weekday()>4)
        df_sess_agg = sessions[['user_id','duration_sec','purchase_amount','purchase_made','is_mobile','is_weekend']].groupby('user_id').agg(['mean','sum']).reset_index()
        df_sess_agg = df_sess_agg[[('user_id',''),('duration_sec','mean'),('purchase_amount','mean'),('purchase_amount','sum'),('purchase_made','mean'),('is_mobile','mean'),('is_weekend','mean')]]
        df_sess_agg.columns = ['user_id','avg_session_duration_sec','avg_session_purchase_amount','total_spend','avg_session_purchase_probability','mobile_usage_pct','weekend_usage_pct']
        df = df.merge(df_sess_agg, how='left', on='user_id').fillna(0)
        
        # ** Features from product_features **
        # feature_adoption_rate
        df_feat = self.features[['user_id','adopted']].groupby('user_id').agg('mean').reset_index()
        df_feat.columns = ['user_id','feature_adoption_rate']
        df = df.merge(df_feat, how='left', on='user_id').fillna(0)

        # ** Features from emails **
        # email_open_pct
        df_email = self.emails[['user_id','opened']].groupby('user_id').agg('mean').reset_index()
        df_email.columns = ['user_id','email_open_pct']
        df = df.merge(df_email, how='left', on='user_id').fillna(0)

        # ** Features from support_tickets **
        # support_tickets
        df_st = self.support_tickets[['user_id','support_ticket_id']].groupby('user_id').agg('count').reset_index()
        df_st.columns = ['user_id','support_tickets']
        df = df.merge(df_st, how='left', on='user_id').fillna(0)
        
        # ** Features from referrals
        # referrals_made, referrals_rate
        df_ref = self.referrals.groupby('user_id').agg('count').reset_index()
        df_ref.columns = ['user_id','referrals_made']
        df = df.merge(df_ref, how='left', on='user_id').fillna(0)
        df['referral_rate'] = df.referrals_made / df.days_since_signup

        self.user_summary = df
        return df 
    
# end

def assign_churn_conversion(df_stats, random_seed=11) -> pd.DataFrame:
    '''
    Assign noisy churn/conversion score to data with stats, for purpose of fitting models for future classification
    
    Arguments:
        - df_stats:  pd.DataFrame with relevant statitics (from get_user_data_statistics)
        
    Returns: pd.DataFrame with 'churn' and 'conversion' score added
    '''
    df = df_stats.copy()
    n_users = len(df)
    np.random.seed(random_seed)

    max_days_since_last_visit = df.days_since_last_visit.max()
    max_sessions_last_30d = df.sessions_last_30d.max()
    max_referral_rate = df.referral_rate.max()
    max_avg_session_duration_sec = df.avg_session_duration_sec.max()

    # ** churn score: high when low engagement and long time since last visit
    churn_score = (
        0.3 * (df.days_since_last_visit / max_days_since_last_visit)
        + 0.2 * (1 - df.feature_adoption_rate)
        + 0.3 * (1 - df.sessions_last_30d / max_sessions_last_30d)
        + 0.2 * (1 - df.referral_rate / max_referral_rate)
        + np.random.normal(scale=0.1, size=n_users)
    )
    df['churned'] = (churn_score > 0.6).astype(int)
    
    
    # ** conversion score: high when high engagement and high feature adoption
    conversion_score = (
        0.2 * df.feature_adoption_rate
        + 0.2 * df.sessions_last_30d / max_sessions_last_30d
        + 0.2 * df.avg_session_duration_sec / max_avg_session_duration_sec
        + 0.2 * df.email_open_pct
        + 0.1 * df.avg_session_purchase_probability
        + 0.1 * df.referral_rate / max_referral_rate
        + np.random.normal(scale=0.1, size=n_users)
    )
    df['converted'] = (conversion_score > 0.6).astype(int)

    return df