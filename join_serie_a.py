import pandas as pd
import os 
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import seaborn as sns
import pycountry_convert as pc   
from collections import defaultdict
import statsmodels.api as sm



def era(s):
    l = {'2009-2013':list(range(2009,2014)),'2014-2017':list(range(2014,2018)),'2017-2021':list(range(2018,2022))}
    for key, years_list in l.items():
        if s in years_list:
            return key
        
def to_row(df,year):
   d=pd.DataFrame(columns=df.columns)
   for col in df.columns:
      c=df[col]
      mean=c.mean()
      d.loc[year,col]=mean
   d=d.to_numpy()
   return d

def plot_correlation_heatmap(df):

    # Calculate the correlation matrix
    corr = df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # Add title
    plt.title('Correlation Heatmap')
    
    # Show plot
    plt.savefig('Heatmap.png',dpi=300)  
    plt.close()

def fail_check(dic,key):
  if (key in dic.keys()):
      return False
  else:
      return True

def create_folder_if_not_exists(directory):

    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' was created.")
    else:
        print(f"Directory '{directory}' already exists.")

def toyear(s):
    try:
      if '/' in s:
        return s.split('/')[-1]
      else:
        return s
    except:
      pass

def country_to_continent(country_name):
      if country_name in ['Bosnia-Herzegovina','England','Kosovo','Northern Ireland','Wales','Scotland']:
         return 'Europe'
      elif country_name in ['DR Congo',"Cote d'Ivoire",'The Gambia']:
         return 'Africa'
      elif country_name in ['Korea, North','Korea, South']:
         return 'Asia'
      else:
        country_code = pc.country_name_to_country_alpha2(country_name)
        continent_code = pc.country_alpha2_to_continent_code(country_code)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name

class Team:
    def __init__(self,name) -> None:
        
        self.name=name

        self.reg_df=pd.DataFrame(columns=['total_cf', 'Total_inflow', 'Total_outflow', 'Foreign_total_cf',
       'Foreign_net_cf', 'foreign-to-total', 'N_foreign_transactions',
       'Perc_foreign_transactions', 'Avg_Age_for_in', 'Avg_Age_for_out'], index=list(np.arange(2009,2018)))

    def players_add(self,dir):
        self.players=pd.read_csv(dir)

class FootballIndustry:
    def __init__(self) -> None:
        self.dictionary=defaultdict(None)
        self.teams=[]
        self.season_stats=defaultdict(None)
        self.colors=['red', 'blue','brown', 'green', 'orange','darkred','yellow', 'grey', 'purple', 
      'pink',  'gold', 'silver', 
      'lime', 'olive', 'chocolate', 'navy', 'teal', 
      'maroon', 'turquoise', 'violet', 'indigo', 'coral', 
      'peachpuff', 'lightblue', 'plum', 'tan', 'lavender','black']
    def load_demographics(self,directory):
        self.directory=directory 
        self.dx=self.directory+'Serie A teams All played players/'
        df=pd.DataFrame(columns=['Name','Country','Position','Born'])
        for filename in os.listdir(self.dx):
          if filename.endswith('.csv') and filename != 'test.csv':
            filepath = os.path.join(self.dx, filename)
            if os.path.isfile(filepath):
              df=pd.concat([df,pd.read_csv(filepath)], axis=0)
        df['Born']=df['Born'].apply(toyear)
        df['Born'].fillna(0, inplace=True)
        df['Born']=df['Born'].astype(int)
        self.demo=df
        self.min_y=int(sorted(self.demo['Born'].unique())[1])
        self.max_y=int(sorted(self.demo['Born'].unique())[-1])
    def load_transfers(self,d):
      df=pd.read_csv(d)
      self.transfers=df
      self.transfers=self.transfers[self.transfers['is_loan']==False]
      self.transfers=self.transfers[self.transfers['is_loan_end']==False]
      self.transfers=self.transfers[self.transfers['is_retired']==False]
      self.transfers.drop(['is_loan','is_loan_end','is_retired'], axis=1, inplace=True)
      self.transfers['transfer_premium']=self.transfers['transfer_fee_amnt']-self.transfers['market_val_amnt']
      self.transfers_grouped=self.transfers.groupby('team_country')
    def load_teams_df(self):
       for year in list(range(2009,2022)):
          df=self.season_stats[year]
          for team in df.index:
             #adds team to dict
             if fail_check(self.dictionary,team):
                self.dictionary[team]=Team(team)
                pointer=self.dictionary[team]
             else:
                pointer=self.dictionary[team]
              #adds line to that team
             pointer.reg_df.loc[year]=df.loc[team].tolist()
    def load_info(self):
       
       s=self.season_stats
       d=pd.DataFrame(index=s.keys(),columns=self.season_stats[2020].columns)
       for year,df in s.items():
          row=to_row(df,year)
          d.loc[year,:]=row
       self.league_stats=d
       plot_correlation_heatmap(self.league_stats.drop(['Total_inflow','Total_outflow'], axis=1).corr())
    def geo_plots(self, n_eras=1):
      hist_list=[]
      years_list=list(range(self.min_y,self.max_y,int((self.max_y-self.min_y)/(n_eras))))

      for i in range(0,len(years_list)-1):
        hist_list.append(  self.demo[ ((self.demo['Born']>years_list[i]) & (self.demo['Born']<=years_list[i+1]))] )

      hist_list.append(self.demo[ (self.demo['Born']>years_list[-1])] ) 
      ###plotting
      base = self.directory + 'hist_nat_'
      folder_name = base + f'{n_eras}_eras/'
      world = gpd.read_file('ne_110m_admin_0_countries_lakes/ne_110m_admin_0_countries_lakes.shp')

      create_folder_if_not_exists(folder_name)

      for i in range(len(years_list)):
          df = hist_list[i]
          country_counts = df['Country'].value_counts().reset_index()
          country_counts.columns = ['Country', 'Count']
          fig_df = world.merge(country_counts, how="left", left_on='NAME',right_on="Country")
          fig_df['Count'] = fig_df['Count'].fillna(0)
          fig_df['Log_Count'] = np.log1p(fig_df['Count'])

          fig, ax = plt.subplots(1, 1, figsize=(15, 10))
          world.boundary.plot(ax=ax)
          fig_df.plot(column='Log_Count', ax=ax, legend=True,
                      legend_kwds={'label': "Logarithm of Number of Players by Country",
                                  'orientation': "horizontal"},
                      cmap='viridis')
          filepath = folder_name + f'{years_list[i]}-{years_list[i]+int((self.max_y-self.min_y)/n_eras)}.png'
          plt.savefig(filepath, dpi=300)
          plt.close(fig) 
    def time_series_percentage_foreign(self):
      ts=pd.DataFrame({'Year': np.arange(self.min_y,league.max_y+1),'Perc_Foreign' :[ 0 for _ in range(len(np.arange(self.min_y,league.max_y+1).tolist())) ]})
      ts.fillna(0, inplace=True)
      default_d1=defaultdict(lambda:0)
      default_d1.update(self.demo[self.demo['Country']!='Italy'].groupby('Born')['Country'].count().to_dict())
      ts['Foreign']=ts['Year'].map(default_d1)
      #ts=ts.merge(s1, how='left', left_index=ts.index, right_index=league.df.index)
      default_d2=defaultdict(lambda:0)
      default_d2.update(self.demo[self.demo['Country']=='Italy'].groupby('Born')['Country'].count().to_dict())
      ts['Italian']=ts['Year'].map(default_d2)
      ts['Total']=ts['Italian']+ts['Foreign']
      ts['Perc_Foreign']=ts['Foreign']/ts['Total']
      ##Trimming
      ts=ts[ts['Year']>1890]
      ts=ts[ts['Year']<1990]
      ts.set_index('Year', inplace=True)
      plt.figure(figsize=(14, 7))  # Set the figure size for better visibility
      plt.plot(ts.index, ts['Perc_Foreign'], label='', color='b', linewidth=2.0)
      ts.drop(list(range(min(ts.index),1945)), axis=0, inplace=True)
      ts.drop(['Foreign','Italian','Total'], inplace=True, axis=1)
      sns.regplot(ts, x=ts.index, y=ts['Perc_Foreign'])
      ##perfroming the regression
      X = sm.add_constant(ts.index)
      # Fit the regression model
      model = sm.OLS(ts,X).fit()
      # Print the summary of the regression model
      print(model.summary())
      plt.savefig(self.directory+'percentage_foreign_ts.png',dpi=300)
    def cash_flow_plot(self, difference=True): 
      plt.figure(figsize=(12, 6))
      ##Color & ticks:
      
      tx=['-', '--', '-.', ':']
      ##Creating TOTAL cf timeseries
      total_cf=self.transfers.groupby(['team_country','season'])['transfer_fee_amnt'].sum().reset_index()
      # Pivot the table to convert it into a time series
      total_cf = total_cf.pivot_table(index='season', columns='team_country', values='transfer_fee_amnt')
      ##
      ##Creating Foreign cf timeseries
      local_cf=self.transfers[self.transfers['team_country'] == self.transfers['counter_team_country'] ].groupby(['team_country','season'])['transfer_fee_amnt'].sum().reset_index()
      
      # Pivot the table to convert it into a time series
      local_cf = local_cf.pivot_table(index='season', columns='team_country', values='transfer_fee_amnt')
      ##
      df_list=[total_cf,local_cf]
      ##
      for d in range(len(df_list)):
        for i in range(len(total_cf.columns)):
            plt.plot(df_list[d].index, df_list[d][df_list[d].columns[i]], label=df_list[d].columns[i], color=self.colors[i], linestyle=tx[d])
      #####
      plt.title('Total and Local Cash flow')
      plt.xlabel('Season')
      plt.ylabel('Cash Flow')
      plt.legend()
      plt.grid(True)
      plt.tight_layout()
      plt.savefig(directory+'total_cash_flow.png',dpi=300)
      plt.close()

      if difference:
        plt.figure(figsize=(12, 6))
        foreign_cf=total_cf-local_cf
        i=0
        for country in foreign_cf.columns:
          plt.plot(foreign_cf.index, foreign_cf[country], label=country, color=self.colors[i])
          i+=1
        plt.title('Foreign Cash flow')
        plt.xlabel('Season')
        plt.ylabel('Cash Flow')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(directory+'foreign_cash_flow.png',dpi=300)
        plt.close()
    def net_flow_plot(self):
        plt.figure(figsize=(12,6))
        sold = self.transfers[
        (self.transfers['dir'] == 'left') &  # Transfers where the direction is 'left'
         (self.transfers['team_country'] != self.transfers['counter_team_country'])  # Different origin and destination countries
        ].groupby(['team_country', 'season'])['transfer_fee_amnt'].sum().reset_index()  # Grouping by country and season, summing the transfer fees

        # Aggregating bought transfers
        bought = self.transfers[
            (self.transfers['dir'] == 'in') &  # Transfers where the direction is 'in'
            (self.transfers['team_country'] != self.transfers['counter_team_country'])  # Different origin and destination countries
        ].groupby(['team_country', 'season'])['transfer_fee_amnt'].sum().reset_index()  # Grouping by country and season, summing the transfer fees
        print(sold)
        print(bought)       
        print()
        sold['net_flow']=sold['transfer_fee_amnt']-bought['transfer_fee_amnt']
        net=sold.drop('transfer_fee_amnt',axis=1)
        net_pivot = net.pivot_table(index='season', columns='team_country', values='net_flow')
        i=0
        for country in net_pivot.columns:
           plt.plot(net_pivot.index, net_pivot[country], label=country, color=self.colors[i])
           i+=1

        plt.title('Net Foreign Cash Flow')
        plt.xlabel('Season')
        plt.ylabel('Net Flow')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(directory+'net_cash_flow.png',dpi=300)
    def partnrship_transactions_heatmap(self):
      ## Creating Folder
        folder_name=self.directory+'transaction_heatmaps'
        try:
          os.makedirs(folder_name, exist_ok=True)
          print(f"Folder '{folder_name}' prepared for file saving.")
        except Exception as e:
          print(f"An error occurred: {e}")
        dataset = self.transfers[self.transfers['counter_team_country'].isin(self.transfers['team_country'].unique())]
        ##Iterating over every season
        season_flag=2009
        while season_flag+4 < 2022:
          print(season_flag)
          l1=season_flag
          l2=season_flag+4
          filename=folder_name+f'/{l1}-{l2}_heatmap.png'
          df=dataset[(dataset['season']>=l1)&(dataset['season']<=l2)] 
          plt.figure(figsize=(12,10)) ##preparing data & figure
          df = df.groupby(['team_country', 'counter_team_country']).size().unstack(fill_value=0)
          mask=np.eye(df.shape[0], dtype=bool)
          sns.heatmap(df, mask=mask,annot=True, cmap='coolwarm', fmt='d') #plotting the heatmap
          #sns.heatmap(np.log1p(df), annot=True, cmap='coolwarm', fmt='.2f') #plotting the heatmap
          plt.title(f'Log-Scaled Transactions Heatmap in {l1}-{l2}')
          plt.ylabel('Team Country')
          plt.xlabel('Counter Team Country')
          plt.savefig(filename, dpi=300)
          plt.close()
          season_flag+=4
    def volume_transaction(self):
        folder_name=self.directory+'volume_transaction/'
        create_folder_if_not_exists(folder_name)
        ##################################################In
        plt.figure(figsize=(12,6))
        df=self.transfers[self.transfers['dir']=='in']
        ##range
        df.loc[:,'season_range']=df['season'].apply(era)
        box_df=df[['team_country','transfer_fee_amnt','season_range']]
        box_df.loc[:,'transfer_fee_amnt']=np.log(box_df['transfer_fee_amnt'])
        ##plotting
        sns.boxplot(box_df,x='team_country',y='transfer_fee_amnt', hue='season_range')
        plt.title('Transfer Fee Amounts of Purchase, by Team Country and Season')
        plt.xlabel('Team Country')
        plt.ylabel('Log Transfer Fee Amount (in Euros)')
        plt.xticks(rotation=45)  # Rotate the x labels for better readability
        plt.legend(title='Season')
        plt.tight_layout()
        plt.savefig(folder_name+'transfer_fee_in.png', dpi=300)
        plt.close()
        ##################################################Out
        plt.figure(figsize=(12,6))
        df=self.transfers[self.transfers['dir']=='left']
        ##range
        df.loc[:,'season_range']=df['season'].apply(era)
        box_df=df[['team_country','transfer_fee_amnt','season_range']]
        box_df.loc[:,'transfer_fee_amnt']=np.log(box_df['transfer_fee_amnt'])
        ##plotting
        sns.boxplot(box_df,x='team_country',y='transfer_fee_amnt', hue='season_range')
        plt.title('Transfer Fee Amounts of Sell, by Team Country and Season')
        plt.xlabel('Team Country')
        plt.ylabel('Log Transfer Fee Amount (in Euros)')
        plt.xticks(rotation=45)  # Rotate the x labels for better readability
        plt.legend(title='Season')
        plt.tight_layout()
        plt.savefig(folder_name+'transfer_fee_out.png', dpi=300)
        plt.close()
    def premium_plot(self):
      ##half the database
      df=self.transfers[~self.transfers['transfer_premium'].isna()]
      ########## Average transfer premium sold
      plot_df=df[df['dir']=='in']
      plot_df.groupby(['team_country','season'])['transfer_premium'].mean().reset_index
      pivot_df=plot_df.pivot_table(index='season',columns='team_country',values='transfer_premium')
      plt.figure(figsize=(12,6))
      i=0
      for country in pivot_df.columns:
         plt.plot(pivot_df.index,pivot_df[country], label=country, color=self.colors[i])
         i+=1
      plt.plot(pivot_df.index, [0 for _ in range(len(pivot_df))], label='X Axis' ,color='black')
      ##plotting
      plt.title('Average Purchase Transfer Premium')
      plt.xlabel('Team Country')
      plt.ylabel('Average transfer premium (in Euros)')
      plt.xticks(rotation=45)  # Rotate the x labels for better readability
      plt.legend(title='Countries')
      plt.tight_layout()
      plt.savefig(self.directory+'average_purchase_premium.png', dpi=300)
      plt.close()

      ########## Average transfer premium sold
      plot_df=df[df['dir']=='left']
      plot_df.groupby(['team_country','season'])['transfer_premium'].mean().reset_index
      pivot_df=plot_df.pivot_table(index='season',columns='team_country',values='transfer_premium')
      plt.figure(figsize=(12,6))
      i=0
      for country in pivot_df.columns:
         plt.plot(pivot_df.index,pivot_df[country], label=country, color=self.colors[i])
         i+=1
      plt.plot(pivot_df.index, [0 for _ in range(len(pivot_df))], label='X Axis' ,color='black')
      ##plotting
      plt.title('Average Selling Transfer Premium')
      plt.xlabel('Team Country')
      plt.ylabel('Average transfer premium (in Euros)')
      plt.xticks(rotation=45)  # Rotate the x labels for better readability
      plt.legend(title='Countries')
      plt.tight_layout()
      plt.savefig(self.directory+'average_selling_premium.png', dpi=300)
      plt.close()

      ########## Average transfer premium sold without Italy
      plot_df=df[df['dir']=='left']
      plot_df=plot_df[plot_df['team_country']!='Italy']
      plot_df.groupby(['team_country','season'])['transfer_premium'].mean().reset_index
      pivot_df=plot_df.pivot_table(index='season',columns='team_country',values='transfer_premium')
      plt.figure(figsize=(12,6))
      i=0
      for country in pivot_df.columns:
         plt.plot(pivot_df.index,pivot_df[country], label=country, color=self.colors[i])
         i+=1
      plt.plot(pivot_df.index, [0 for _ in range(len(pivot_df))], label='X Axis' ,color='black')
      ##plotting
      plt.title('Average Selling Transfer Premium')
      plt.xlabel('Team Country')
      plt.ylabel('Average transfer premium (in Euros)')
      plt.xticks(rotation=45)  # Rotate the x labels for better readability
      plt.legend(title='Countries')
      plt.tight_layout()
      plt.savefig(self.directory+'average_selling_premium_noitaly.png', dpi=300)
      plt.close()     
    def age_plot(self):
      pivot_means = pd.pivot_table(self.transfers, values='player_age', index=['team_country', 'season'], columns='dir', aggfunc='mean')

      # Now, calculate the difference between 'in' and 'left' ages
      pivot_means['age_difference'] = pivot_means['in'] - pivot_means['left']

      # Resetting the index to make 'team_country' and 'season' into columns again
      pivot_df = pivot_means['age_difference'].unstack(level=-1)  # Unstack seasons to make them columns
      fig, ax = plt.subplots(figsize=(10, 6))
      i=0
      for i, country in enumerate(pivot_df.index):
          seasons = pivot_df.columns.astype(str)  # Convert season to string if necessary
          ax.plot(seasons, pivot_df.loc[country], label=country, color=self.colors[i], marker='o')

      ax.axhline(0, color='black', linewidth=0.8) 
      ##plotting
      ax.set_title('Age Difference by Season for Each Country')
      ax.set_xlabel('Season')
      ax.set_ylabel('Mean Age Difference (In - Left)')
      ax.legend(title='Team Country')
      plt.tight_layout()
      plt.savefig(self.directory+'average_age_change.png', dpi=300)
      plt.close()

      
# Assuming the data loading and preprocessing part is correct
directory='/Users/emanuelesebastianelli/Desktop/industrial_economics_project/'
league=FootballIndustry()
league.load_demographics(directory)
league.load_transfers('/Users/emanuelesebastianelli/Desktop/industrial_economics/transfers.csv')
#league.geo_plots(n_eras=6)
#league.net_flow_plot()
#league.partnrship_transactions_heatmap()
league.volume_transaction()
#league.premium_plot()
#league.age_plot()